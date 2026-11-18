"""Derive the capture entry point without changing the published timing harness."""
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
text = source.read_text().replace('from forward_probe import install', 'from capture_probe import install')
start = text.index("    reports = [json.loads(")
end = text.index("\nif __name__ == '__main__':", start)
text = text[:start] + '''    reports = [json.loads((out / f'rank-{r}.json').read_text()) for r in range(gpu_count)]
    assert all(r['finite'] and r['capture_complete'] for r in reports)
    summary = dict(capture_complete=True, timing_samples=0,
                   trajectory='dense', denoise_step=0,
                   terminal_exception=terminal_exception,
                   saved_layers=reports[0]['saved_layers'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\\n')
    print('CAPTURE_SUMMARY', json.dumps(summary), flush=True)
''' + text[end:]
text = text.replace('Paired same-process Dense cuDNN vs upstream SOL; alternating order; 3 warmup pairs + 9 measured pairs; max rank wall latency',
                    'Untimed first conditional 50-layer dense DiT forward; selected attention inputs captured after Ulysses')
target.write_text(text)
