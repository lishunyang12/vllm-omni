#!/usr/bin/env python3
"""Reproduce shared pooling precision on an already configured CUDA environment.

Example: bash reproduce_pool_fp32.sh --workdir /tmp/empty-sol-pool-check
No GPU allocation, clock changes, weights, or CUDA visibility overrides.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

REVISIONS = {"old": "cf67a3acad7511acb85053e56a05e5dd24c82ba0", "fixed": "0f0d973354fa9dc003de51e38b530e73f42df003"}
PACKAGE = "techniques/sparse_backends/sol_attn/"
TEST = "techniques/sparse_backends/tests/test_preprocess_precision.py"
TEST_SHA = "6ea93154137cdd4cfbf398ba1a66cd680df6d96a2335fafaa5b2e15d0448d13d"
CHANGES = {
    PACKAGE + "preprocess.py": [
        ("tl.sum(values, axis=0) / block_len", "tl.sum(values.to(tl.float32), axis=0) / block_len"),
        ("summary = tl.sum(values, axis=0)\n", "summary = tl.sum(values.to(tl.float32), axis=0)\n"),
    ],
    PACKAGE + "triton_ref/preprocess.py": [
        ("tl.sum(k_values, axis=0)", "tl.sum(k_values.to(tl.float32), axis=0)"),
        ("tl.sum(v_values, axis=0)", "tl.sum(v_values.to(tl.float32), axis=0)"),
    ],
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def fetch(root, label, revision):
    url = f"https://codeload.github.com/NVlabs/Sana/tar.gz/{revision}"
    archive = root / f"{label}.tar.gz"
    checksum = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("xb") as stream:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > 256 * 1024 * 1024:
                raise RuntimeError("Source archive exceeded 256 MiB limit")
            checksum.update(block)
            stream.write(block)
    manifest, retained = {}, {}
    prefix = None
    expanded = 0
    destination = root / label
    destination.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            prefix = path.parts[0] if prefix is None else prefix
            if path.parts[0] != prefix:
                raise RuntimeError("Archive contains multiple top-level roots")
            if member.isdir():
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative == "." or relative in manifest:
                raise RuntimeError(f"Invalid/duplicate archive entry: {relative}")
            selected = relative.startswith(PACKAGE) or relative == TEST
            if member.issym():
                if selected:
                    raise RuntimeError("Refusing symlink in extracted package/test")
                manifest[relative] = {"kind": "symlink", "target": member.linkname, "mode": member.mode}
                continue  # Compare other repository symlinks, never extract them.
            if not member.isfile() or member.size > 128 * 1024 * 1024:
                raise RuntimeError(f"Unsupported or oversized archive entry: {relative}")
            expanded += member.size
            if expanded > 1024 * 1024 * 1024:
                raise RuntimeError("Expanded source archive exceeded 1 GiB limit")
            with tar.extractfile(member) as source:
                data = source.read()
            manifest[relative] = {"kind": "file", "sha256": digest(data), "mode": member.mode}
            if selected:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if relative in CHANGES or relative == TEST:
                    retained[relative] = data
    return {"revision": revision, "url": url, "archive_sha256": checksum.hexdigest(), "files": manifest}, retained


def run(command, root, env, log, timeout):
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def classify(xml, code, old):
    if not xml.exists():
        return {"classification": "harness_error", "returncode": code}
    cases = ET.parse(xml).getroot().findall(".//testcase")
    failures = [failure for case in cases for failure in case.findall("failure")]
    errors = sum(len(case.findall("error")) for case in cases)
    skipped = sum(len(case.findall("skipped")) for case in cases)
    valid = len(cases) == 12 and errors == 0 and skipped == 0
    if old:
        valid = (
            valid
            and code == 1
            and bool(failures)
            and all("vs float64" in "".join(failure.itertext()) for failure in failures)
        )
        status = "expected_numeric_failure" if valid else "unexpected_old_result"
    else:
        valid = valid and code == 0 and not failures
        status = "expected_pass" if valid else "fixed_failed"
    return dict(
        classification=status, returncode=code, tests=len(cases), failures=len(failures), errors=errors, skipped=skipped
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path, help="A nonexistent or empty directory")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--install-pytest", action="store_true", help="Explicitly permit installing pytest==8.3.5 only")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()) or args.timeout_s <= 0:
        parser.error("workdir must be empty and timeout positive")
    env = os.environ.copy()
    report = {
        "valid": False,
        "python": sys.version,
        "executable": sys.executable,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "environment": {
            key: env.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "TRITON_CACHE_DIR",
            )
        },
        "sources": {},
        "tests": {},
    }
    output = root / "report.json"
    save(output, report)
    try:
        retained = {}
        for label, revision in REVISIONS.items():
            report["sources"][label], retained[label] = fetch(root, label, revision)
            save(output, report)
        old, fixed = [report["sources"][label]["files"] for label in ("old", "fixed")]
        changed = {key for key in old.keys() | fixed.keys() if old.get(key) != fixed.get(key)}
        assert changed == set(CHANGES) | {TEST}, changed
        assert TEST not in old and fixed[TEST]["sha256"] == TEST_SHA
        for name, replacements in CHANGES.items():
            expected = retained["old"][name].decode()
            for before, after in replacements:
                assert expected.count(before) == 1, (name, before)
                expected = expected.replace(before, after)
            assert expected.encode() == retained["fixed"][name], name
        report["verified_change"] = (
            "Exactly four FP32 input casts plus pinned regression test; other archive entries identical"
        )
        if args.install_pytest:
            code = run(
                [sys.executable, "-m", "pip", "install", "pytest==8.3.5"],
                root,
                env,
                root / "install.log",
                args.timeout_s,
            )
            if code:
                raise RuntimeError("Explicit pytest installation failed")
        probe = (
            "import json,os,torch,triton,pytest; "
            "assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0]>=9; "
            "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,"
            "pytest=pytest.__version__,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),"
            "CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))))"
        )
        if run([sys.executable, "-c", probe], root, env, root / "environment.json", args.timeout_s):
            raise RuntimeError(
                "Pinned CUDA/Triton/pytest environment or TMA-capable GPU is unavailable; see environment.json"
            )
        report["runtime"] = json.loads((root / "environment.json").read_text().splitlines()[-1])
        test = root / "test_preprocess_precision.py"
        test.write_bytes(retained["fixed"][TEST])
        for label in ("old", "fixed"):
            child_env = dict(
                env,
                PYTHONPATH=str(root / label / "techniques/sparse_backends"),
                PYTHONPYCACHEPREFIX=str(root / "pycache" / label),
                PYTHONDONTWRITEBYTECODE="1",
                CUTE_DSL_DISABLE_FILE_CACHING="1",
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            )
            xml = root / f"{label}.xml"
            code = run(
                [sys.executable, "-m", "pytest", str(test), "-q", f"--junitxml={xml}"],
                root,
                child_env,
                root / f"{label}.log",
                args.timeout_s,
            )
            report["tests"][label] = classify(xml, code, label == "old")
            save(output, report)
        report["valid"] = (
            report["tests"]["old"]["classification"] == "expected_numeric_failure"
            and report["tests"]["fixed"]["classification"] == "expected_pass"
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(output, report)
    print(json.dumps({"report": str(output), "valid": report["valid"], "tests": report["tests"]}, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
