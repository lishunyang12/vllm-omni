#!/usr/bin/env python3
"""Serialized attention benchmark with one original-name source per process.

The controller imports no CUDA libraries. Workers inherit CUDA_VISIBLE_DEVICES
unchanged, import exactly one sol_attn source (or forced cuDNN Dense), and wait
for synchronous JSON commands. All hashes/checks are outside timed regions.
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
import platform
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_CACHE_DISABLE",
    "CUTE_DSL_DISABLE_FILE_CACHING",
    "CUTE_DSL_ARCH",
    "CUTE_DSL_KEEP_PTX",
    "CUTE_DSL_KEEP_CUBIN",
    "TRITON_CACHE_DIR",
    "TRITON_ALWAYS_COMPILE",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_CPUS_PER_TASK",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPYCACHEPREFIX",
)
NAMES = ("baseline", "candidate", "dense")


def environment():
    return {key: os.environ.get(key) for key in ENV_KEYS}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_manifest(package):
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if path.is_file() and "__pycache__" not in relative.parts and path.suffix not in (".pyc", ".pyo"):
            files[relative.as_posix()] = file_hash(path)
    if not files or "__init__.py" not in files:
        raise ValueError(f"Invalid source package: {package}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"package": str(package), "files": files, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def require_fp32_pooling(package):
    """Require the shared explicit-FP32 pooling fix before any CUDA import."""
    expected = {
        "preprocess.py": {"_reduce_kc_kernel": 1, "_reduce_vc_kernel": 1},
        "triton_ref/preprocess.py": {"_reduce_kv_kernel": 2},
    }
    hashes = {}
    for relative, functions in expected.items():
        path = package / relative
        tree = ast.parse(path.read_text())
        for name, count in functions.items():
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = [
                node for node in ast.walk(function) if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.sum"
            ]
            if len(calls) != count or not all(
                (
                    isinstance(call.args[0], ast.Call)
                    and isinstance(call.args[0].func, ast.Attribute)
                    and call.args[0].func.attr == "to"
                    and len(call.args[0].args) == 1
                    and ast.unparse(call.args[0].args[0]) == "tl.float32"
                )
                or any(
                    keyword.arg == "dtype" and ast.unparse(keyword.value) == "tl.float32" for keyword in call.keywords
                )
                for call in calls
            ):
                raise RuntimeError(f"Shared FP32 pooling fix required before benchmarking: {path}:{name}")
        hashes[relative] = file_hash(path)
    return {"explicit_fp32_required": True, "source_sha256": hashes}


def rotated_order(index):
    offset = index % len(NAMES)
    order = list(NAMES[offset:] + NAMES[:offset])
    if index % 2:
        order.reverse()
    return order


class Worker:
    """Created only after the controller sends init; no background GPU work."""

    def __init__(self, name, config):
        if os.environ.get("CUTE_DSL_DISABLE_FILE_CACHING") != "1":
            raise RuntimeError("CUTE_DSL_DISABLE_FILE_CACHING must be 1 before import")
        self.name = name
        self.package = None
        self.manifest = None
        self.pooling_contract = None
        self.mode = "eager"
        self.graph = self.graph_output = self.graph_stream = None
        self.last_output = None
        self.source_root = config.get("source_root")
        if "sol_attn" in sys.modules:
            raise RuntimeError("sol_attn was imported before selecting the worker source")
        if name != "dense":
            self.package = Path(self.source_root).resolve() / "techniques/sparse_backends/sol_attn"
            self.manifest = source_manifest(self.package)
            self.pooling_contract = require_fp32_pooling(self.package)
            sys.path.insert(0, str(self.package.parent))
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("An existing CUDA allocation is required")
        self.device = torch.cuda.current_device()
        if torch.cuda.get_device_capability(self.device) not in ((10, 0), (10, 3)):
            raise RuntimeError("This benchmark requires SM100/SM103")
        self.sdpa_kernel, self.cudnn_backend = sdpa_kernel, SDPBackend.CUDNN_ATTENTION
        self.sol = None
        self.prepared_values = None
        self.prepared_calls = 0
        self.preprocess = None
        if name != "dense":
            self.sol = importlib.import_module("sol_attn").sol_attn
            self.preprocess = importlib.import_module("sol_attn.preprocess")
            import triton

            # Same pooling launch configuration on both sides: precision and work match.
            for kernel in (self.preprocess._reduce_kc_kernel, self.preprocess._reduce_vc_kernel):
                kernel.configs = [triton.Config({}, num_warps=4, num_stages=1)]
                kernel.cache.clear()
            self.native_prepare = self.preprocess.prepare
            self.preprocess.prepare = self.record_prepare
            self.check_imports()

        payload = torch.load(config["input"], map_location="cpu", weights_only=True)
        expected = payload["tensor_sha256"]
        if not {"q", "k", "v"} <= set(expected):
            raise ValueError("Payload must record full q/k/v tensor digests")
        full_hashes = {}
        for key, target in expected.items():
            full_hashes[key] = self.digest(payload[key])
            if full_hashes[key] != target:
                raise ValueError(f"Frozen payload digest failed: {key}")
        heads = config["heads"]
        selected = []
        for key in ("q", "k", "v"):
            value = payload[key]
            if value.ndim != 4 or value.shape[2] < heads or value.shape[3] != 128 or value.dtype != torch.bfloat16:
                raise ValueError(f"Invalid BTHD BF16 tensor or insufficient heads: {key}")
            selected.append(value[:, :, :heads].contiguous())
        if len({tuple(value.shape) for value in selected}) != 1:
            raise ValueError("Q/K/V shape mismatch")
        self.q, self.k, self.v = [value.to("cuda") for value in selected]
        cpu_mask = payload["exact_block_mask"]
        self.mask = None if cpu_mask is None else cpu_mask.to("cuda").contiguous()
        if self.mask is not None:
            if self.mask.dtype != torch.bool or tuple(self.mask.shape) != ((self.q.shape[1] + 63) // 64,):
                raise ValueError("Invalid exact_block_mask")
        self.options = dict(payload["options"])
        if "exact_block_mask" in self.options:
            raise ValueError("Mask must be stored separately from payload options")
        self.input_metadata = {
            "input": str(Path(config["input"]).resolve()),
            "full_tensor_sha256": full_hashes,
            "shape": list(self.q.shape),
            "dtype": str(self.q.dtype),
            "heads_selected": heads,
            "options": self.options,
            "denoise_step": payload.get("denoise_step"),
            "layer": payload.get("layer"),
            "trajectory": payload.get("trajectory"),
            "protected_blocks": None if self.mask is None else int(self.mask.sum()),
        }
        del payload, selected, cpu_mask, value
        self.inputs = {"q": self.q, "k": self.k, "v": self.v}
        if self.mask is not None:
            self.inputs["exact_block_mask"] = self.mask
        self.input_hashes = {key: self.digest(value) for key, value in self.inputs.items()}
        self.input_metadata["selected_tensor_sha256"] = self.input_hashes
        props = torch.cuda.get_device_properties(self.device)
        self.runtime = {
            "pid": os.getpid(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "environment": environment(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "capability": list(torch.cuda.get_device_capability(self.device)),
            "device_index_within_inherited_visibility": self.device,
            "total_memory": props.total_memory,
            "sm_count": props.multi_processor_count,
            "torch_config": torch.__config__.show(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "backend": "CUDNN_ATTENTION (forced, no fallback)" if name == "dense" else "sol_attn",
            "pooling_contract": self.pooling_contract,
        }
        if name == "dense":
            functional = Path(torch.nn.functional.__file__).resolve()
            self.runtime["torch_functional_source"] = {"path": str(functional), "sha256": file_hash(functional)}
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            self.last_output = self.invoke()
            torch.cuda.synchronize()
        self.initial = self.snapshot()

    def digest(self, value):
        raw = value.detach().contiguous().cpu().view(self.torch.uint8).numpy()
        return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()

    def check_imports(self):
        if self.package is None:
            return {}
        origins = {}
        for name, module in sorted(sys.modules.items()):
            if name == "sol_attn" or name.startswith("sol_attn."):
                origin = getattr(module, "__file__", None)
                if origin:
                    resolved = Path(origin).resolve()
                    if not resolved.is_relative_to(self.package):
                        raise RuntimeError(f"Wrong-source import: {name} -> {resolved}")
                    origins[name] = str(resolved)
        if "sol_attn" not in origins:
            raise RuntimeError("Original-name sol_attn was not loaded")
        return origins

    def invoke(self):
        if self.name == "dense":
            with self.sdpa_kernel(self.cudnn_backend):
                return self.torch.nn.functional.scaled_dot_product_attention(
                    self.q.transpose(1, 2),
                    self.k.transpose(1, 2),
                    self.v.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.options.get("scale"),
                ).transpose(1, 2)
        return self.sol(self.q, self.k, self.v, **self.options, exact_block_mask=self.mask)

    def record_prepare(self, *args, **kwargs):
        # Keep exactly the values consumed by the native attention call. During
        # graph capture these are graph-owned buffers updated by every replay.
        # No digest, CPU copy or synchronization occurs in this timed wrapper.
        values = self.native_prepare(*args, **kwargs)
        self.prepared_values = values
        self.prepared_calls += 1
        return values

    def snapshot(self):
        torch = self.torch
        torch.cuda.synchronize()
        hashes = {key: self.digest(value) for key, value in self.inputs.items()}
        immutable = hashes == self.input_hashes
        finite = bool(torch.isfinite(self.last_output).all())
        output = {
            "sha256": self.digest(self.last_output),
            "shape": list(self.last_output.shape),
            "dtype": str(self.last_output.dtype),
            "finite": finite,
        }
        current_manifest = None if self.package is None else source_manifest(self.package)
        source_unchanged = current_manifest == self.manifest
        prepared = None
        autotune = None
        if self.name != "dense":
            if self.prepared_values is None or len(self.prepared_values) != 3:
                raise RuntimeError("Native prepare was not observed; cannot certify identical routing inputs")
            prepared = {
                key: {
                    "sha256": self.digest(value),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value).all()),
                }
                for key, value in zip(("kc", "vc", "threshold"), self.prepared_values)
            }
            if not all(item["finite"] for item in prepared.values()):
                raise RuntimeError("Non-finite prepared attention input")
            autotune = {
                key: str(getattr(getattr(self.preprocess, key), "best_config", None))
                for key in (
                    "_reduce_kc_kernel",
                    "_reduce_vc_kernel",
                    "_reduce_kc_stats_kernel",
                    "_diag_threshold_kernel",
                )
            }
        result = {
            "input_sha256": hashes,
            "input_immutable": immutable,
            "output": output,
            "source_unchanged": source_unchanged,
            "loaded_module_origins": self.check_imports(),
            "mode": self.mode,
            "prepared": prepared,
            "prepared_strategy": "retain actual native prepare outputs; end-to-end preprocessing remains timed",
            "prepare_python_calls": self.prepared_calls,
            "preprocess_autotune": autotune,
        }
        if not immutable or not finite or not source_unchanged:
            raise RuntimeError(f"Worker validation failed: {json.dumps(result)}")
        return result

    def prepare(self, mode, capture_warmup):
        torch = self.torch
        torch.cuda.synchronize()
        self.graph = self.graph_output = self.graph_stream = None
        self.mode = mode
        setup = {"mode": mode, "capture_eager_warmup": 0, "capture_calls": 0, "validation_replays": 0}
        with torch.inference_mode():
            if mode == "graph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    for _ in range(capture_warmup):
                        self.last_output = self.invoke()
                    stream.synchronize()
                    with torch.cuda.graph(graph, stream=stream):
                        output = self.invoke()
                torch.cuda.current_stream().wait_stream(stream)
                self.graph, self.graph_output, self.graph_stream = graph, output, stream
                graph.replay()
                torch.cuda.synchronize()
                self.last_output = output
                setup.update(capture_eager_warmup=capture_warmup, capture_calls=1, validation_replays=1)
        return setup

    def run(self, measured):
        torch = self.torch
        torch.cuda.synchronize()
        unix_ns, monotonic_ns = time.time_ns(), time.perf_counter_ns()
        with torch.inference_mode():
            if measured:
                self.begin.record()
            if self.mode == "graph":
                self.graph.replay()
                output = self.graph_output
            else:
                output = self.invoke()
            if measured:
                self.end.record()
            torch.cuda.synchronize()
        wall_ns = time.perf_counter_ns() - monotonic_ns
        self.last_output = output
        return {
            "mode": self.mode,
            "measured": measured,
            "start_unix_s": unix_ns / 1e9,
            "start_unix_ns": unix_ns,
            "start_monotonic_ns": monotonic_ns,
            "wall_s": wall_ns / 1e9,
            "event_ms": self.begin.elapsed_time(self.end) if measured else None,
        }


def worker_main(name):
    # Preserve a dedicated protocol descriptor; all Python/native stdout from
    # Torch/CuTe/compiler subprocesses is redirected to the stderr log.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    instance = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            command = request["cmd"]
            if command == "init":
                if instance is not None:
                    raise RuntimeError("Worker already initialized")
                instance = Worker(name, request["config"])
                result = {
                    "variant": name,
                    "runtime": instance.runtime,
                    "source": instance.manifest,
                    "input": instance.input_metadata,
                    "initial": instance.initial,
                    "initial_untimed_calls": 1,
                }
            elif command == "shutdown":
                result = {"shutdown": True}
            elif instance is None:
                raise RuntimeError("Initialize before GPU commands")
            elif command == "prepare":
                result = instance.prepare(request["mode"], request["capture_warmup"])
            elif command == "run":
                result = instance.run(request["measured"])
            elif command == "check":
                result = instance.snapshot()
            else:
                raise ValueError(f"Unknown command: {command}")
            protocol.write(json.dumps({"id": request["id"], "ok": True, "result": result}, allow_nan=False) + "\n")
            protocol.flush()
            if command == "shutdown":
                return 0
        except BaseException as error:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, flush=True)
            protocol.write(
                json.dumps(
                    {
                        "id": request.get("id"),
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": detail,
                    }
                )
                + "\n"
            )
            protocol.flush()
            return 1
    return 0


class Client:
    def __init__(self, name, args, logs, cache_root):
        self.name, self.timeout = name, args.timeout_s
        self.buffer, self.sequence = b"", 0
        self.log_path = logs / f"{name}.stderr.log"
        self.log = self.log_path.open("xb")
        env = os.environ.copy()
        env["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # An empty per-worker pycache location also prevents reading stale
        # bytecode from a source checkout; source names stay sol_attn.
        env["PYTHONPYCACHEPREFIX"] = str(cache_root / name)
        try:
            self.process = subprocess.Popen(
                [args.python, "-u", str(Path(__file__).resolve()), "--worker", name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                env=env,
                start_new_session=True,
                bufsize=0,
            )
        except BaseException:
            self.log.close()
            raise
        os.set_blocking(self.process.stdout.fileno(), False)

    def request(self, command, **values):
        self.sequence += 1
        request = {"id": self.sequence, "cmd": command, **values}
        raw = (json.dumps(request, allow_nan=False) + "\n").encode()
        self.process.stdin.write(raw)
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.name}/{command} exceeded {self.timeout}s; {self.log_path}")
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                continue
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"{self.name} exited during {command}; {self.log_path}")
            self.buffer += data
            if len(self.buffer) > 16 * 1024 * 1024:
                raise RuntimeError(f"Oversized worker protocol response: {self.name}")
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if response.get("id") != self.sequence:
            raise RuntimeError(f"Out-of-order worker response: {self.name}")
        if not response.get("ok"):
            raise RuntimeError(f"{self.name}/{command}: {response.get('error')}; {self.log_path}")
        return response["result"]

    def kill(self):
        # Kill the process group, including any compiler subprocess, even if
        # the worker already exited. Never leave another worker active on error.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait_and_close(self):
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.kill()
            self.process.wait(timeout=10)
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()


def validate_checks(checks, initial, previous=None):
    expected_inputs = initial["baseline"]["input_sha256"]
    for name, check in checks.items():
        if (
            check["input_sha256"] != expected_inputs
            or not check["input_immutable"]
            or not check["source_unchanged"]
            or not check["output"]["finite"]
        ):
            raise RuntimeError(f"Input/source/finite gate failed: {name}")
        if previous is not None and check["output"] != previous[name]["output"]:
            raise RuntimeError(f"Output changed over the timed interval: {name}")
    for name in ("baseline", "candidate"):
        if checks[name]["prepared"] is None:
            raise RuntimeError(f"Missing prepared-input evidence: {name}")
        if checks[name]["prepared"] != initial["baseline"]["prepared"]:
            raise RuntimeError(f"KC/VC/threshold byte hashes differ across sources or timed modes: {name}")
        if previous is not None and checks[name]["prepared"] != previous[name]["prepared"]:
            raise RuntimeError(f"Prepared inputs changed over the timed interval: {name}")
        if checks[name]["output"] != initial["baseline"]["output"]:
            raise RuntimeError(f"SOL output changed across modes/warmup: {name}")
    if checks["baseline"]["output"] != checks["candidate"]["output"]:
        raise RuntimeError("Baseline/candidate output byte hashes, shapes or dtypes differ")


def controller(args):
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report}")
    logs = args.report.parent / (args.report.stem + ".workers")
    logs.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "valid": False,
        "protocol": "one source per resident process; serialized JSON RPC",
        "script_sha256": file_hash(Path(__file__).resolve()),
        "parent_environment": environment(),
        "input": str(args.input.resolve()),
        "warmup_rounds_per_mode": args.warmup,
        "measured_rounds_per_mode": args.rounds,
        "graph_setup_eager_warmup": args.warmup,
        "heads": args.heads,
        "timeout_s_per_command": args.timeout_s,
        "workers": {},
        "modes": {},
        "error": None,
        "correctness_scope": (
            "SOL output and actual KC/VC/threshold byte identity to the supplied baseline; "
            "Dense hash is recorded separately. Native preprocessing remains included in all timings. "
            "This is not an independent numerical oracle."
        ),
    }
    save(args.report, report)
    clients = {}
    success = False
    with tempfile.TemporaryDirectory(prefix="sol-process-bench-pycache-") as temporary:
        try:
            for name in NAMES:
                clients[name] = Client(name, args, logs, Path(temporary))
            # Only this controller issues commands; initialization and every
            # GPU command finish before any command is sent to another worker.
            for name in NAMES:
                source = args.baseline if name == "baseline" else args.candidate
                config = {
                    "input": str(args.input.resolve()),
                    "heads": args.heads,
                    "source_root": None if name == "dense" else str(source.resolve()),
                }
                metadata = clients[name].request("init", config=config)
                report["workers"][name] = metadata
                if metadata["runtime"]["environment"]["CUDA_VISIBLE_DEVICES"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
                    raise RuntimeError("Worker changed inherited CUDA_VISIBLE_DEVICES")
                save(args.report, report)
            initial = {name: report["workers"][name]["initial"] for name in NAMES}
            contracts = [report["workers"][name]["runtime"]["pooling_contract"] for name in ("baseline", "candidate")]
            if contracts[0] != contracts[1] or not contracts[0]["explicit_fp32_required"]:
                raise RuntimeError("Baseline/candidate must use identical shared FP32 pooling sources")
            devices = {
                (item["runtime"]["device_index_within_inherited_visibility"], item["runtime"]["gpu_uuid"])
                for item in report["workers"].values()
            }
            if len(devices) != 1:
                raise RuntimeError("Workers did not select the same inherited GPU")
            validate_checks(initial, initial)
            for mode in args.modes.split(","):
                entry = {"setup": {}, "warmup_order": [], "before": {}, "samples": [], "after": {}, "metrics": {}}
                report["modes"][mode] = entry
                for name in NAMES:
                    entry["setup"][name] = clients[name].request("prepare", mode=mode, capture_warmup=args.warmup)
                for iteration in range(args.warmup):
                    order = rotated_order(iteration)
                    entry["warmup_order"].append(order)
                    for name in order:
                        clients[name].request("run", measured=False)
                entry["before"] = {name: clients[name].request("check") for name in NAMES}
                validate_checks(entry["before"], initial)
                save(args.report, report)
                for iteration in range(args.rounds):
                    order = rotated_order(iteration)
                    for position, name in enumerate(order):
                        sample = clients[name].request("run", measured=True)
                        sample.update(round=iteration, variant=name, order=order, position=position)
                        entry["samples"].append(sample)
                        save(args.report, report)
                entry["after"] = {name: clients[name].request("check") for name in NAMES}
                validate_checks(entry["after"], initial, previous=entry["before"])
                for name in NAMES:
                    values = [sample["event_ms"] for sample in entry["samples"] if sample["variant"] == name]
                    entry["metrics"][name] = {
                        "median_ms": statistics.median(values),
                        "mean_ms": statistics.mean(values),
                        "cv_percent": 100 * statistics.stdev(values) / statistics.mean(values),
                        "samples": len(values),
                    }
                entry["validated"] = True
                save(args.report, report)
                print(mode, json.dumps(entry["metrics"]), file=sys.stderr, flush=True)
            for client in clients.values():
                client.request("shutdown")
            for client in clients.values():
                if client.process.wait(timeout=10) != 0:
                    raise RuntimeError(f"Worker exited nonzero: {client.name}")
            success = True
        except BaseException as error:
            report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            raise
        finally:
            if not success:
                for client in clients.values():
                    client.kill()
            cleanup_errors = []
            for client in clients.values():
                try:
                    client.wait_and_close()
                except BaseException as error:
                    cleanup_errors.append(f"{client.name}: {error}")
            report["cleanup_errors"] = cleanup_errors
            report["child_exit_codes"] = {name: client.process.returncode for name, client in clients.items()}
            report["complete"] = success and not cleanup_errors
            report["valid"] = report["complete"]
            save(args.report, report)
            if success and cleanup_errors:
                raise RuntimeError(f"Child cleanup failed: {cleanup_errors}")
    print(json.dumps({"complete": True, "valid": True, "report": str(args.report)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.worker:
        return worker_main(args.worker)
    if any(getattr(args, key) is None for key in ("baseline", "candidate", "input", "report")):
        parser.error("--baseline, --candidate, --input and --report are required")
    modes = args.modes.split(",")
    if len(modes) != len(set(modes)) or not set(modes) <= {"eager", "graph"}:
        parser.error("--modes must contain distinct eager/graph values")
    if args.warmup < 1 or args.rounds < 2 or args.heads < 1 or args.timeout_s <= 0:
        parser.error("Require warmup>=1, rounds>=2, heads>=1, timeout-s>0")

    def interrupted(signum, _frame):
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    controller(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
