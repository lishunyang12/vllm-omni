"""One untimed call to export fresh compiler artifacts for resource inspection."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from sol_attn import sol_attn, interface

from ablation import digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
data = torch.load(a.input, map_location='cpu', weights_only=True)
for key, value in data['tensor_sha256'].items():
    assert digest(data[key]) == value
with torch.inference_mode():
    q, k, v = [data[n][:, :, :28].contiguous().cuda() for n in ('q','k','v')]
    result = sol_attn(q, k, v, **data['options'], exact_block_mask=data['exact_block_mask'].cuda())
    torch.cuda.synchronize()
    assert bool(torch.isfinite(result).all())
root = Path(interface.__file__).parent
a.output.write_text(json.dumps(dict(shape=list(q.shape), capability=list(torch.cuda.get_device_capability()),
    torch=torch.__version__, source_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest()
        for n in ('interface.py','sm100/mainloop.py','sm100/softmax.py')}, timing_samples=0),indent=2)+'\n')
