from typing import Any


def resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return
    worker_cls = engine_args.get("worker_cls")
    if worker_cls is not None and worker_cls != "auto":
        return
    from vllm_omni.platforms import current_omni_platform

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")
