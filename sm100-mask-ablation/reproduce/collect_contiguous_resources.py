"""CPU artifact inspection only: cuobjdump resources and static PTX opcode sites."""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cuobjdump", required=True)
    args = parser.parse_args()
    report = {
        "scope": "Static generated artifact counts only; no dynamic execution counts or occupancy inference",
        "shared_memory_limit": "cuobjdump resource SHARED does not include launch-time dynamic shared memory",
        "branch_limit": (
            "Candidate retains fallback branches: total static TMA sites may increase "
            "even when its eligible path issues fewer copies"
        ),
        "variants": {},
    }
    metadata = {}
    for variant in ("baseline", "candidate"):
        folder = args.root / variant
        metadata[variant] = json.loads((folder / "meta.json").read_text())
        cubins, ptxs = sorted(folder.rglob("*.cubin")), sorted(folder.rglob("*.ptx"))
        assert cubins and ptxs, f"Missing fresh compiler artifacts: {variant}"
        resources, instructions = [], []
        for path in cubins:
            result = subprocess.run(
                [args.cuobjdump, "--dump-resource-usage", str(path)], check=True, capture_output=True, text=True
            )
            raw = result.stdout + result.stderr
            (path.parent / (path.name + ".resources.txt")).write_text(raw)
            resources.append(
                dict(
                    file=str(path.relative_to(folder)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    command=[args.cuobjdump, "--dump-resource-usage", str(path)],
                    output=raw,
                )
            )
        for path in ptxs:
            source = path.read_text()
            code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
            code = re.sub(r"//[^\n]*", "", code)
            opcodes = Counter(re.findall(r"^\s*(?:@!?%[\w$]+\s+)?([a-z][\w]*(?:\.[\w:]+)*)\b", code, re.M))
            counts = {
                "branch_sites": sum(n for op, n in opcodes.items() if op == "bra" or op.startswith("bra.")),
                "tma_tensor_copy": sum(n for op, n in opcodes.items() if op.startswith("cp.async.bulk.tensor.")),
                "mbarrier_expect_tx": sum(
                    n for op, n in opcodes.items() if op.startswith("mbarrier.") and "expect_tx" in op
                ),
                "mbarrier_wait": sum(
                    n
                    for op, n in opcodes.items()
                    if op.startswith("mbarrier.") and ("try_wait" in op or "test_wait" in op)
                ),
                "tcgen05_mma": sum(n for op, n in opcodes.items() if op.startswith("tcgen05.mma.")),
                "tcgen05_commit": sum(n for op, n in opcodes.items() if op.startswith("tcgen05.commit.")),
                "tcgen05_wait": sum(n for op, n in opcodes.items() if op.startswith("tcgen05.wait.")),
                "bar_sync": sum(n for op, n in opcodes.items() if op == "bar.sync" or op.startswith("barrier.")),
            }
            instructions.append(
                dict(
                    file=str(path.relative_to(folder)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    static_sites=counts,
                    all_opcode_sites=dict(sorted(opcodes.items())),
                )
            )
        report["variants"][variant] = dict(resources=resources, ptx=instructions)
    baseline, candidate = metadata["baseline"], metadata["candidate"]
    report["input_prepared_output_match"] = all(
        baseline[key] == candidate[key] for key in ("input_sha256", "prepared", "output_sha256", "pooling_config")
    )
    (args.root / "resource-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["input_prepared_output_match"], "Sources did not consume/produce identical fixed-pool values"
    print(args.root / "resource-comparison.json")


if __name__ == "__main__":
    main()
