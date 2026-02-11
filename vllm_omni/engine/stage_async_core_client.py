"""
Stage Async Engine Core Client for vLLM-Omni V1 architecture.

Directly inherits from vLLM's AsyncMPClient to reuse EngineCore architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient

from vllm_omni.engine.stage_init import StageMetadata

if TYPE_CHECKING:
    from vllm.inputs import TextPrompt
    from vllm.v1.engine import EngineCoreOutput
    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageAsyncCoreClient(AsyncMPClient):
    """Stage async client that inherits from vLLM's AsyncMPClient.

    Fully reuses AsyncMPClient.__init__ for:
    - ZMQ setup, sockets
    - launch_core_engines() -> EngineCoreProc
    - outputs_queue, output_queue_task
    - All utility methods (shutdown, get_output_async, abort_requests_async, etc.)

    This is the async version of StageMPClient, designed for use with AsyncOmniEngine.
    """

    def __init__(
        self,
        vllm_config: Any,
        executor_class: type,
        metadata: StageMetadata,
    ):
        """Create an async EngineCore client for a single stage.

        All heavy init (config extraction, plugin loading, device setup,
        engine args building, device locking) is done by the Orchestrator
        via helpers in stage_init.py.  This constructor just stores metadata
        and calls super().__init__().
        """
        # -------- Stage metadata (public fields used at runtime) --------
        self.stage_id = metadata.stage_id
        self.stage_type = metadata.stage_type
        self.engine_output_type = metadata.engine_output_type
        self.is_comprehension = metadata.is_comprehension
        self.requires_multimodal_data = metadata.requires_multimodal_data
        self.engine_input_source = metadata.engine_input_source
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.custom_process_input_func = metadata.custom_process_input_func
        self.model_stage = metadata.model_stage

        self.engine_outputs: Any = None

        logger.info(
            "[StageAsyncCoreClient] Stage-%s initializing EngineCore",
            self.stage_id,
        )
        super().__init__(vllm_config, executor_class, log_stats=False)
        logger.info(
            "[StageAsyncCoreClient] Stage-%s EngineCore running",
            self.stage_id,
        )

    # ==================== Overrides ====================

    async def add_request_async(
        self, request: EngineCoreRequest | dict[str, Any]
    ) -> None:
        """Add request - supports both EngineCoreRequest and task dict."""
        logger.info(
            f"[StageAsyncCoreClient] Stage-{self.stage_id} adding request: {request.request_id if isinstance(request, EngineCoreRequest) else request.get('request_id', 'N/A')}"
        )
        await super().add_request_async(request)

    # ==================== Stage Methods ====================

    def set_engine_outputs(self, engine_outputs: "EngineCoreOutput") -> None:
        """Set engine outputs (called by orchestrator)."""
        self.engine_outputs = engine_outputs

    def process_engine_inputs(
        self,
        stage_list: list[Any],
        prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    ) -> list["OmniTokensPrompt | TextPrompt"]:
        """Process inputs from upstream stages."""
        from vllm_omni.inputs.data import OmniTokensPrompt

        if self.custom_process_input_func is not None:
            logger.info(
                f"[StageAsyncCoreClient] Stage-{self.stage_id} using custom process input function"
            )
            return self.custom_process_input_func(
                stage_list,
                self.engine_input_source,
                prompt,
                self.requires_multimodal_data,
            )

        if not self.engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self.stage_id}")

        source_id = self.engine_input_source[0]
        source_outputs = stage_list[source_id].engine_outputs

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {
            so.request_id: p.get("multi_modal_data")
            for so, p in zip(source_outputs, prompt)
        }

        # logger.info(
        #     f"[StageAsyncCoreClient] Stage-{self.stage_id} processing engine inputs: {source_outputs}"
        # )
        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=(
                    mm_data[so.request_id] if self.requires_multimodal_data else None
                ),
            )
            for so in source_outputs
        ]
