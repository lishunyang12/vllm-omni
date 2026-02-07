## Purpose

Fix a server crash (`AssertionError: seqlen_ro must be >= seqlen`) in the Qwen-Image-Edit online serving pipeline when users provide explicit `height`/`width` in `extra_body` for image editing with a non-square input image.

**Root cause:** In `pre_process_func`, the input image was resized to the user-specified `(height, width)` (e.g., 1024x1024), but the aspect-ratio-preserving `calculated_height`/`calculated_width` (e.g., 1056x992) were stored in `additional_information`. Downstream, the pipeline builds `img_shapes` from both:

- `img_shapes[0]` = `(1, 64, 64)` from explicit 1024x1024 (noise latent)
- `img_shapes[1]` = `(1, 66, 62)` from calculated 1056x992 (image latent)

The image latent's actual seq_len (4096) exceeds the RoPE seq_len computed from `calculated_height/width` (4092), triggering the assertion failure.

**Fix:** Resize and preprocess the input image using `calculated_height`/`calculated_width` instead of the explicit `height`/`width`, matching the fallback/debug path behavior (line 665). This ensures `img_shapes[1]` is always consistent with the actual image latent dimensions.

Also fixes the Python example client where `--height`/`--width` CLI arguments were accepted but never sent to the server.

## Test Plan

- Restart the server with Qwen-Image-Edit model
- Test with non-square input image + explicit `height`/`width` via curl: should succeed instead of crashing

```bash
bash examples/online_serving/image_to_image/run_curl_image_edit.sh input.png "Convert to watercolor style"
```

- Test with Python client (no explicit dims): should still work as before

```bash
python examples/online_serving/image_to_image/openai_chat_client.py \
    --input input.png --prompt "Convert to watercolor style"
```

- Test with Python client (explicit dims): should work now that `height`/`width` are actually sent

```bash
python examples/online_serving/image_to_image/openai_chat_client.py \
    --input input.png --prompt "Convert to watercolor style" \
    --height 1024 --width 1024
```

## Test Result

Before this fix, the following request with a non-square image (e.g., 514x556) crashes the server:

```
AssertionError: seqlen_ro must be >= seqlen
```

After this fix, the same request completes successfully.

## Changes

| File | Change |
|------|--------|
| `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py` | Resize input image to `calculated_height/width` instead of explicit `height/width` in `pre_process_func` |
| `examples/online_serving/image_to_image/run_curl_image_edit.sh` | Remove hardcoded `height`/`width` from `extra_body` to let the server auto-compute from input |
| `examples/online_serving/image_to_image/openai_chat_client.py` | Wire `height`/`width` into `extra_body` so CLI args actually take effect |

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft.
</details>
