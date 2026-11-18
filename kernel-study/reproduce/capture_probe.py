"""Capture real H3 attention inputs in a separate, untimed dense forward."""
import hashlib
import json
import os
from pathlib import Path


def install():
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnImpl
    from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
    from vllm_omni.diffusion.forward_context import get_forward_context
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    original = MiniMaxH3DiTModel.forward
    if getattr(original, '_h3_capture', False):
        return

    def capture(self, *args, **kwargs):
        root = Path(os.environ['H3_FORWARD_RESULT'])
        if not (root / 'armed').exists():
            return original(self, *args, **kwargs)
        if get_forward_context().denoise_step_idx != 0 or len(self.blocks) != 50:
            raise RuntimeError('Expected first conditional forward of all 50 H3 layers')
        selected = {int(v) for v in os.environ.get('H3_CAPTURE_LAYERS', '0,12,25,49').split(',')}
        rank = dist.get_rank()
        rows = []
        saved = []
        prior = SolAttnImpl.forward_cuda

        def observe(impl, query, key, value, attn_metadata=None):
            used = impl._used_length(query, key, value, attn_metadata)
            q, k, v = (x[:, :used].contiguous() for x in (query, key, value))
            mask = impl._sink_block_mask(attn_metadata, used, q.device)
            sink_start, sink_tokens = impl._sink_range(attn_metadata, used)
            layer = impl.layer_idx
            row = dict(layer=layer, shape=list(q.shape), used=used)
            if layer in selected and rank == 0:
                payload = dict(q=q.cpu(), k=k.cpu(), v=v.cpu(),
                               exact_block_mask=None if mask is None else mask.cpu(),
                               options=dict(scale=impl.softmax_scale, tau=impl.config.tau,
                                            thresh_type=impl.config.thresh_type,
                                            sink_start=sink_start, sink_tokens=sink_tokens,
                                            kv_splits=impl._resolve_kv_splits(used)),
                               layer=layer, rank=rank, trajectory='dense', denoise_step=0)
                hashes = {}
                for name in ('q', 'k', 'v', 'exact_block_mask'):
                    t = payload[name]
                    if t is not None:
                        hashes[name] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                payload['tensor_sha256'] = hashes
                name = f'qkv-layer-{layer:02d}-rank-{rank}.pt'
                torch.save(payload, root / name)
                row.update(file=name, tensor_sha256=hashes, options=payload['options'])
                saved.append(layer)
                print('H3_QKV_CAPTURE', json.dumps(row), flush=True)
            rows.append(row)
            dense = CuDNNAttentionImpl(q.shape[2], 128, impl.softmax_scale)
            output = dense.forward_cuda(q, k, v)
            if used < query.shape[1]:
                output = torch.cat((output, torch.zeros_like(query[:, used:])), dim=1)
            return output

        SolAttnImpl.forward_cuda = observe
        try:
            result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            SolAttnImpl.forward_cuda = prior
        if len(rows) != 50 or (rank == 0 and set(saved) != selected):
            raise RuntimeError(f'Incomplete capture: {len(rows)} layers, saved {saved}')
        finite = all(bool(torch.isfinite(t).all()) for t in result)
        if not finite:
            raise RuntimeError('Nonfinite dense output')
        report = dict(rank=rank, world_size=dist.get_world_size(), finite=finite,
                      capture_complete=True, saved_layers=saved, layers=rows,
                      gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      trajectory='dense', denoise_step=0, timing_samples=0)
        (root / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        dist.barrier()
        raise RuntimeError('H3_QKV_CAPTURE_COMPLETE: intentional stop before sampler and VAE decode')

    capture._h3_capture = True
    MiniMaxH3DiTModel.forward = capture
