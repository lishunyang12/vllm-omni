"""Dense H3 trajectory with frozen captures and one paired step-10 forward."""
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
    from step10_forward_fp32 import install as install_bench

    original = MiniMaxH3DiTModel.forward
    if getattr(original, '_sequence_probe', False):
        return
    install_bench()
    benchmark = MiniMaxH3DiTModel.forward
    seen = set()
    trajectory_bytes_match = False

    def sequence(self, *args, **kwargs):
        nonlocal trajectory_bytes_match
        root = Path(os.environ['H3_FORWARD_RESULT'])
        if not (root / 'armed').exists():
            return original(self, *args, **kwargs)
        step = get_forward_context().denoise_step_idx
        rank = dist.get_rank()
        sparse = [m.attention for m in self.modules() if hasattr(m, 'attention') and isinstance(m.attention, SolAttnImpl)]
        if len(sparse) != 50:
            raise RuntimeError(f'Expected 50 H3 layers, got {len(sparse)}')
        for impl in sparse:
            impl.dense_layers = set(range(50))
            impl.dense_attention = CuDNNAttentionImpl(1, 128, impl.softmax_scale)
        if step not in (0, 10, 25) or step in seen:
            print(f'DENSE_TRAJECTORY rank={rank} step={step}', flush=True)
            return original(self, *args, **kwargs)
        prior = SolAttnImpl.forward_cuda
        captured = []
        rows = []

        def observe(impl, query, key, value, attn_metadata=None):
            used = impl._used_length(query, key, value, attn_metadata)
            q, k, v = (x[:, :used].contiguous() for x in (query, key, value))
            if impl.layer_idx in (0, 25) and rank == 0:
                mask = impl._sink_block_mask(attn_metadata, used, q.device)
                start, count = impl._sink_range(attn_metadata, used)
                payload = dict(q=q.cpu(), k=k.cpu(), v=v.cpu(), exact_block_mask=mask.cpu(),
                               options=dict(scale=impl.softmax_scale, tau=impl.config.tau,
                                            thresh_type=impl.config.thresh_type, sink_start=start,
                                            sink_tokens=count, kv_splits=impl._resolve_kv_splits(used)),
                               layer=impl.layer_idx, rank=rank, trajectory='dense', denoise_step=step)
                payload['tensor_sha256'] = {n: hashlib.sha256(payload[n].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                            for n in ('q','k','v','exact_block_mask')}
                path = root / f'qkv-step-{step:02d}-layer-{impl.layer_idx:02d}-rank-{rank}.pt'
                torch.save(payload, path)
                captured.append(impl.layer_idx)
                print('CAPTURED', path.name, flush=True)
            rows.append(impl.layer_idx)
            output = CuDNNAttentionImpl(q.shape[2], 128, impl.softmax_scale).forward_cuda(q, k, v)
            if used < query.shape[1]:
                output = torch.cat((output, torch.zeros_like(query[:, used:])), dim=1)
            return output

        SolAttnImpl.forward_cuda = observe
        try:
            result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            SolAttnImpl.forward_cuda = prior
        assert len(rows) == 50 and (rank != 0 or captured == [0,25])
        assert all(bool(torch.isfinite(t).all()) for t in result)
        seen.add(step)
        (root/f'capture-step-{step:02d}-rank-{rank}.json').write_text(json.dumps(dict(
            denoise_step=step, rank=rank, layers=rows, captured=captured, trajectory='dense',
            shape_capture='after Ulysses exchange', finite=True),indent=2)+'\n')
        if step == 10:
            measured_dense = benchmark(self, *args, **kwargs)
            assert len(measured_dense) == len(result) == 2
            for actual, expected in zip(measured_dense, result):
                torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8), atol=0, rtol=0)
            trajectory_bytes_match = True
            print('STEP10_DENSE_TRAJECTORY_BYTES_MATCH', rank, flush=True)
        if step == 25:
            report = dict(rank=rank, capture_complete=True, finite=True, captured_steps=sorted(seen), step10_dense_bytes_match=trajectory_bytes_match)
            (root/f'rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
            dist.barrier()
            raise RuntimeError('H3_SEQUENCE_CAPTURE_COMPLETE: intentional stop before step-25 sampler advancement and VAE decode')
        return result

    sequence._sequence_probe = True
    MiniMaxH3DiTModel.forward = sequence
