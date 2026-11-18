"""Profile one warmed original/candidate kernel on a hash-checked H3 capture."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from sol_attn import sol_attn
from benchmark_sm100 import baseline

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['baseline', 'candidate'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
data = torch.load('/work/captures/qkv-layer-00-rank-0.pt', weights_only=True, map_location='cpu')
for name in ('q', 'k', 'v', 'exact_block_mask'):
    value = data[name]
    if value is not None:
        assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == data['tensor_sha256'][name]
q, k, v = [data[n][:, :, :28].contiguous().cuda() for n in ('q', 'k', 'v')]
mask = data['exact_block_mask'].cuda()
with tempfile.TemporaryDirectory(prefix='sol-profile-baseline-') as directory, torch.inference_mode():
    old, hashes = baseline(Path('/work/source/baseline/techniques/sparse_backends'), Path(directory))
    fn = old if a.variant == 'baseline' else sol_attn
    for _ in range(3):
        out = fn(q, k, v, **data['options'], exact_block_mask=mask)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('H3_SOL_PROFILE')
    out = fn(q, k, v, **data['options'], exact_block_mask=mask)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    a.output.write_text(json.dumps(dict(variant=a.variant, shape=list(q.shape), baseline_sha256=hashes, capture_sha256=data['tensor_sha256'], finite=bool(torch.isfinite(out).all())), indent=2)+'\n')
