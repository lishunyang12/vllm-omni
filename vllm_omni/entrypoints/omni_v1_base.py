from __future__ import annotations

import os
import time
import types
from collections.abc import Iterable, Sequence
from pprint import pformat
from typing import Any, Literal

from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _dummy_snapshot_download(model_id: str) -> str:
    return model_id


def omni_snapshot_download(model_id: str) -> str:
    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)
    return _dummy_snapshot_download(model_id)


OutputMessageHandleResult = (
    tuple[Literal[True], None, None, None]
    | tuple[Literal[False], str, int, ClientRequestState]
)


class OmniV1Base:
    """Shared runtime foundation for AsyncOmniV1 and OmniV1."""

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        log_stats: bool = False,
        async_chunk: bool = False,
        output_modalities: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if "log_requests" in kwargs:
            raise TypeError("`log_requests` has been removed in OmniV1/AsyncOmniV1. Use `log_stats`.")
        model = omni_snapshot_download(model)
        self.model = model
        self.log_stats = log_stats
        self.async_chunk = async_chunk
        self.output_modalities = output_modalities or []

        logger.info("[%s] Initializing with model %s", self.__class__.__name__, model)
        st = time.time()
        self.engine = AsyncOmniEngine(
            model=model,
            stage_configs=stage_configs,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=stage_init_timeout,
            **kwargs,
        )
        et = time.time()
        logger.info("[%s] AsyncOmniEngine initialized in %.2f seconds", self.__class__.__name__, et - st)
        self.async_chunk = bool(self.async_chunk or getattr(self.engine, "async_chunk", False))

        self.request_states: dict[str, ClientRequestState] = {}

        self.default_sampling_params_list = self.engine.default_sampling_params_list
        if not self.output_modalities:
            self.output_modalities = [
                self.engine.get_stage_metadata(i).get("final_output_type") for i in range(self.engine.num_stages)
            ]

        self._stage_meta_list = [
            types.SimpleNamespace(**self.engine.get_stage_metadata(i)) for i in range(self.engine.num_stages)
        ]

        logger.info(
            "[%s] Initialized with %s stages for model %s",
            self.__class__.__name__,
            self.engine.num_stages,
            model,
        )

    @property
    def num_stages(self) -> int:
        return self.engine.num_stages

    @property
    def errored(self) -> bool:
        return hasattr(self.engine, "orchestrator_thread") and not self.engine.orchestrator_thread.is_alive()

    @property
    def is_running(self) -> bool:
        return not self.errored

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    def check_health(self) -> None:
        if self.errored:
            raise EngineDeadError("Orchestrator process is not alive")

    def resolve_sampling_params_list(
        self,
        sampling_params_list: Sequence[Any] | Any | None,
    ) -> Sequence[Any]:
        if sampling_params_list is None or not isinstance(sampling_params_list, Sequence):
            normalized = self.default_sampling_params_list
        else:
            normalized = sampling_params_list
        if len(normalized) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} sampling params, got {len(normalized)}"
            )
        return normalized

    def _create_request_state(
        self,
        request_id: str,
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> ClientRequestState:
        metrics = OrchestratorMetrics(
            self.num_stages,
            self.log_stats,
            wall_start_ts,
            final_stage_id_for_e2e,
        )
        req_state = ClientRequestState(request_id)
        req_state.metrics = metrics
        self.request_states[request_id] = req_state
        return req_state

    def submit_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_list: Sequence[Any],
        final_stage_id_for_e2e: int,
        wall_start_ts: float,
    ) -> tuple[ClientRequestState, float]:
        req_state = self._create_request_state(
            request_id=request_id,
            wall_start_ts=wall_start_ts,
            final_stage_id_for_e2e=final_stage_id_for_e2e,
        )
        self.engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id_for_e2e,
        )
        submit_ts = time.time()
        if req_state.metrics is not None:
            req_state.metrics.stage_first_ts[0] = submit_ts
        return req_state, submit_ts

    def _log_summary_and_cleanup(self, request_id: str) -> None:
        req_state = self.request_states.get(request_id)
        try:
            if req_state is None or req_state.metrics is None:
                return
            summary = req_state.metrics.build_and_log_summary()
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception:
            logger.exception(
                "[%s] Failed to build/log summary for req=%s",
                self.__class__.__name__,
                request_id,
            )
        finally:
            self.request_states.pop(request_id, None)

    def _abort_requests(self, request_id: str | Iterable[str]) -> list[str]:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        if not request_ids:
            return []
        self.engine.abort(request_ids)
        for req_id in request_ids:
            self.request_states.pop(req_id, None)
        if self.log_stats:
            logger.info("[%s] Aborted request(s) %s", self.__class__.__name__, ",".join(request_ids))
        return request_ids

    def _compute_final_stage_id(self, output_modalities: list[str] | None) -> int:
        return get_final_stage_id_for_e2e(
            output_modalities,
            self.output_modalities,
            self._stage_meta_list,
        )

    def _process_stage_metrics_message(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("request_id")
        req_state = self.request_states.get(req_id)
        if req_state is None or req_state.metrics is None:
            return
        _m = msg.get("metrics")
        if _m is None:
            return
        stage_id = msg.get("stage_id", 0)
        req_state.metrics.on_stage_metrics(stage_id, req_id, _m)
        submit_ts = msg.get("stage_submit_ts")
        now = time.time()
        if req_state.metrics.stage_first_ts[stage_id] is None:
            req_state.metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        req_state.metrics.stage_last_ts[stage_id] = max(req_state.metrics.stage_last_ts[stage_id] or 0.0, now)

    def _handle_output_message(
        self,
        msg: dict[str, Any] | None,
    ) -> OutputMessageHandleResult:
        """Handle one Orchestrator output-queue message."""
        if msg is None:
            return True, None, None, None

        msg_type = msg.get("type")
        if msg_type == "stage_metrics":
            self._process_stage_metrics_message(msg)
            return True, None, None, None

        if msg_type == "error":
            raise RuntimeError(msg.get("error", "Orchestrator returned an error message"))

        if msg_type != "output":
            logger.warning("[%s] got unexpected msg type: %s", self.__class__.__name__, msg_type)
            return True, None, None, None

        req_id = msg.get("request_id")
        if req_id is None:
            logger.warning("[%s] got output message without request_id", self.__class__.__name__)
            return True, None, None, None

        stage_id = msg.get("stage_id")
        if stage_id is None:
            logger.warning("[%s] got output message without stage_id for req=%s", self.__class__.__name__, req_id)
            return True, None, None, None

        req_state = self.request_states.get(req_id)
        if req_state is None:
            logger.debug(
                "[%s] dropping output for unknown req %s",
                self.__class__.__name__,
                req_id,
            )
            return True, None, None, None

        req_state.stage_id = stage_id

        return False, req_id, stage_id, req_state

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> OmniRequestOutput | None:
        req_id = result.get("request_id")
        engine_outputs = result.get("engine_outputs")
        finished = engine_outputs.finished

        submit_ts = result.get("stage_submit_ts")
        now = time.time()
        if metrics.stage_first_ts[stage_id] is None:
            metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, now)

        _m = result.get("metrics")
        if finished and _m is not None:
            metrics.on_stage_metrics(stage_id, req_id, _m)

        stage_meta = self.engine.get_stage_metadata(stage_id)
        if not stage_meta["final_output"]:
            return None

        try:
            rid_key = str(req_id)
            if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                metrics.on_finalize_request(
                    stage_id,
                    req_id,
                    req_start_ts.get(req_id, wall_start_ts),
                )
        except Exception:
            logger.exception("[%s] Finalize request handling error", self.__class__.__name__)

        images = getattr(engine_outputs, "images", []) if stage_meta["final_output_type"] == "image" else []
        return OmniRequestOutput(
            stage_id=stage_id,
            final_output_type=stage_meta["final_output_type"],
            request_output=engine_outputs,
            images=images,
        )

    def shutdown(self) -> None:
        logger.info("[%s] Shutting down", self.__class__.__name__)
        self._shutdown_base()

    def close(self) -> None:
        self.shutdown()

    def _shutdown_base(self) -> None:
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        self.engine.shutdown()
