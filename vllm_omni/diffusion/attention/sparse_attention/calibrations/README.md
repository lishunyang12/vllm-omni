# Skip-Softmax calibrations

Developer-curated, hosted in-repo (not in the checkpoint). Each file is a ModelOpt
`sparse_attention_config` block: `{a, b}` curve + `ignore` list + `target_sparsity`.
Users load the **raw** checkpoint and pass `target_sparsity`; the matching calibration
is resolved via `registry.py`. `target_sparsity` on an unlisted model raises.

| model | files | source | calibrated at |
|-------|-------|--------|---------------|
| Wan 2.2 T2V A14B | `wan2_2_a14b/{transformer,transformer_2}.json` | official (vllm-omni-rankings) | 480x832 / 81f |
