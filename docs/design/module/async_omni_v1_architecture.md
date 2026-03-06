# AsyncOmni V1 Architecture (Qwen3-Omni Example)

## 1. System Architecture

```text
• ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                                    API Layer                                    │
  │  ┌─────────────────────────────────────┐  ┌──────────────────────────────────┐  │
  │  │ AsyncOmni (EngineClient)            │  │ Omni                             │  │
  │  │ • generate() / abort() / shutdown() │  │ • generate()                     │  │
  │  │ • _final_output_handler()           │  │                                  |  │
  │  └─────────────────────────────────────┘  └──────────────────────────────────┘  │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │                              Engine Layer (Proxy)                               │
  │  ┌───────────────────────────────────────────────────────────────────────────┐  │
  │  │ AsyncOmniEngine                                                           │  │
  │  │ • _bootstrap_orchestrator() & _initialize_stages()                        │  │
  │  │ • add_request() / add_request_async() -> input_processor.process_inputs() │  │
  │  │ • try_get_output() / try_get_output_async()                               │  │
  │  └───────────────────┬─────────────────────────────────▲─────────────────────┘  │
  │         request_queue (janus.Queue)        output_queue (janus.Queue)           │
  ├──────────────────────┼─────────────────────────────────┼────────────────────────┤
  │                      ▼        Orchestration Layer      │                        │
  │  ┌───────────────────────────────────────────────────────────────────────────┐  │
  │  │ Orchestrator [background thread]                                          │  │
  │  │ • _request_handler()                                                      │  │
  │  │     -  stage_client.add_request_async() & _prewarm_async_chunk_stages()   │  │
  │  │ • _orchestration_output_handler()                                         │  │
  │  │     -  _process_stage_outputs() -> output_processors[i].process_outputs() │  │
  │  │     -  _route_output() & _forward_to_next_stage()                         │  │
  │  └──────────┬─────────────────────────┬────────────────────────┬─────────────┘  │
  ├─────────────┼─────────────────────────┼────────────────────────┼────────────────┤
  │             │                 Communication Layer              │                │
  │  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
  │  │ StageEngineCoreClient │ │ StageEngineCoreClient │ │ StageDiffusionClient  │  │
  │  │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │ │ • ZMQ ROUTER / PULL   │  │
  │  │ • Msgpack codec       │ │ • Msgpack codec       │ │ • Msgpack codec       │  │
  │  └──────────┬────────────┘ └──────────┬────────────┘ └──────────┬────────────┘  │
  │             ▼ ZMQ IPC                 ▼ ZMQ IPC                 ▼ ZMQ IPC       │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │                                 Execution Layer                                 │
  │  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐  │
  │  │ StageCoreProc         │ │ StageCoreProc         │ │ DiffusionEngine       │  │
  │  │ [background process]  │ │ [background process]  │ │ [background process]  │  │
  │  └───────────────────────┘ └───────────────────────┘ └───────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Execution Flow (Arrow Steps, one generate request)

```text
[1] App
    -> AsyncOmniV1.generate(prompt, request_id)

[2] AsyncOmniV1
    -> _final_output_handler()   (started on first request)
    -> AsyncOmniEngine.add_request(stage_id=0, ...)

[3] AsyncOmniEngine.add_request
    -> (if stage-0 is llm and input is not EngineCoreRequest)
       InputProcessor.process_inputs()
       OutputProcessor[0].add_request()
    -> request_queue.put(add_request_msg)

[4] Orchestrator._request_handler
    -> _handle_add_request(msg)
    -> stage_clients[0].add_request_async(...)

[5] Orchestrator._orchestration_loop (loop)
    -> poll stage output
       - llm stage: await get_output_async()
       - diffusion stage: get_diffusion_output_async()
    -> (llm stage) output_processors[i].process_outputs(...)
    -> _route_output(...)
    -> if finished and not final_stage and non-async-chunk:
         _forward_to_next_stage(...)
         -> next_stage.add_request_async(...)
    -> output_queue.put(output)

[6] AsyncOmniV1._final_output_loop (background coroutine)
    -> AsyncOmniEngine.try_get_output_blocking()
    -> route by request_id to ClientRequestState.queue

[7] AsyncOmniV1._process_orchestrator_results
    -> read from ClientRequestState.queue
    -> _process_single_result(...)
    -> yield OmniRequestOutput

[8] Exit condition
    -> receive result["finished"] == True
    -> generate() ends
```

## 3. Runtime Sequence (one generate request)

```mermaid
sequenceDiagram
    participant APP as App
    participant V1 as AsyncOmniV1
    participant ENG as AsyncOmniEngine
    participant ORCH as Orchestrator
    participant S0 as Stage-0 Client
    participant SN as Next Stage Client

    APP->>V1: generate
    V1->>V1: start output_handler once
    V1->>ENG: add_request(stage_id=0, ...)
    ENG->>ENG: input_processor.process_inputs()
    ENG->>ORCH: request_queue.put(add_request)

    ORCH->>ORCH: _handle_add_request
    ORCH->>S0: add_request_async

    loop poll route forward
        ORCH->>S0: get_output_async / get_diffusion_output_async
        ORCH->>ORCH: _route_output
        alt need forward to next stage
            ORCH->>SN: add_request_async
        end
        ORCH-->>ENG: output_queue.put
    end

    V1->>ENG: try_get_output_blocking
    ENG-->>V1: message
    V1-->>APP: yield OmniRequestOutput
```

## 4. Comparison

V0: 
```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Main Process                                                               │
│  ┌──────────────────────┐   ┌────────────────────────────────────────────┐ │
│  │ generate()           │   │ final_output_handler()                     │ │
│  └──────────────────────┘   └────────────────────────────────────────────┘ │
└──────────┬─────────────────────────┬─────────────────────────┬─────────────┘
  mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)
           ▼▲                        ▼▲                        ▼▲
┌───────────────────────┐  ┌───────────────────────┐  ┌──────────────────────┐
│ Worker Proc-0         │  │ Worker Proc-1         │  │ Worker Proc-2        │
│ (Thinker LLM)         │  │ (Talker LLM)          │  │ (Vocoder)            │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │_stage_worker   │   │  │  │_stage_worker   │   │  │  │_stage_worker   │  │
│  │_async()        │   │  │  │_async()        │   │  │  │_async()        │  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │output_handler()│   │  │  │output_handler()│   │  │  │output_handler()│  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
└──────────┬────────────┘  └──────────┬────────────┘  └──────────┬───────────┘
       ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ EngineCore Proc-0    │   │ EngineCore Proc-1    │   │ EngineCore Proc-2    │
│ (Thinker)            │   │ (Talker)             │   │ (Vocoder)            │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

V1:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Main Process                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Main Thread                                                          │  │
│  │  ┌──────────────────────┐   ┌─────────────────────────────────────┐  │  │
│  │  │ generate()           │   │ final_output_handler()              │  │  │
│  │  └──────────────────────┘   └─────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         asyncio.Queue (request_queue) ▼  ▲ asyncio.Queue (output_queue)    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Orchestrator Thread                                                  │  │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐  │  │
│  │  │ _request_handler()   │  │ _orchestration_output_handler()      │  │  │
│  │  └──────────────────────┘  └──────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │ engine_core_output_handler()*3  (stage-0 / stage-1 / stage-2)  │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └───────┬─────────────────────────┬─────────────────────────┬──────────┘  │
└──────────┬─────────────────────────┬─────────────────────────┬─────────────┘
       ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ              
  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
  │ EngineCore Proc-0    │  │ EngineCore Proc-1    │  │ EngineCore Proc-2    │
  │ (Thinker)            │  │ (Talker)             │  │ (Vocoder)            │
  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```


Test scripts:
```bash
# enter offline inference folder.
cd qwen2_5_omni

# lagacy impl:
VLLM_OMNI_USE_V1=0 VLLM_LOGGING_LEVEL=INFO python end2end_v1.py --output-wav output_audio \
                  --query-type use_mixed_modalities
# current impl:
VLLM_OMNI_USE_V1=1 VLLM_LOGGING_LEVEL=INFO python end2end_v1.py --output-wav output_audio \
                  --query-type use_mixed_modalities

cd qwen3_omni
# lagacy impl:
VLLM_OMNI_USE_V1=0 python end2end_v1.py --output-wav output_audio --query-type text --async-chunk --enable-stats
# current impl:
VLLM_OMNI_USE_V1=1 python end2end_v1.py --output-wav output_audio --query-type text --async-chunk --enable-stats

cd bagel
# lagacy impl:
VLLM_OMNI_USE_V1=0 python end2end_v1.py --prompts "A cute cat"
# current impl:
VLLM_OMNI_USE_V1=1 python end2end_v1.py --prompts "A cute cat"

cd text_to_image
# lagacy impl:
VLLM_OMNI_USE_V1=0 python text_to_image_async.py --prompt "a cup of coffee on the table" --output output.png
# current impl:
VLLM_OMNI_USE_V1=1 python text_to_image_async.py --prompt "a cup of coffee on the table" --output output.png
```