"""Diagnostic shadow calls; return the reference to keep downstream inputs fixed."""
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def install_shadow():
    import vllm_omni.diffusion.attention.ops.sol_attn as ops
    path = '/work/source/precise/vllm_omni/diffusion/attention/ops/sol_residual.py'
    spec = importlib.util.spec_from_file_location('precise_residual', path)
    precise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precise)
    original = ops._merge_sol_bsa_residual
    index = 0
    rows = []
    captured = False

    def compare(actual, expected):
        diff = actual.float() - expected.float()
        return dict(bitwise_equal=bool(torch.equal(actual, expected)), max_abs=diff.abs().max().item(),
                    relative_l2=(diff.norm()/expected.float().norm()).item(),
                    different_fraction=(actual != expected).float().mean().item())

    def shadow(*args):
        nonlocal index, captured
        expected = original(*args)
        if index < 50:
            rank = dist.get_rank()
            row = dict(layer=index, allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       input_shape=list(args[0].shape), input_dtype=str(args[0].dtype))
            for chunk in [256, 2048]:
                actual = precise.merge_sol_residual(*args, chunk_size=chunk)
                row[str(chunk)] = compare(actual, expected)
            root = Path(os.environ['H3_FORWARD_RESULT'])
            if not captured and not row['2048']['bitwise_equal']:
                if rank == 0:
                    torch.save({'args': [t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in args],
                                'layer': index}, root/'first-mismatch-input.pt')
                captured = True
            rows.append(row)
            (root/f'shadow-rank-{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('RESIDUAL_SHADOW', rank, json.dumps(row), flush=True)
        index += 1
        return expected

    ops._merge_sol_bsa_residual = shadow
