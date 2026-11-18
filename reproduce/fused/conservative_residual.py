import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _merge_residual(residual, exact_output, exact_weights, output, num_elements: tl.constexpr, chunk_len: tl.constexpr, start_query, num_heads: tl.constexpr, head_dim: tl.constexpr, exact_stride_b: tl.constexpr, exact_stride_q: tl.constexpr, exact_stride_h: tl.constexpr, exact_stride_d: tl.constexpr, out_stride_b: tl.constexpr, out_stride_q: tl.constexpr, out_stride_h: tl.constexpr, out_stride_d: tl.constexpr, block_size: tl.constexpr):
    ids = tl.program_id(0) * block_size + tl.arange(0, block_size)
    d = ids % head_dim
    h = ids // head_dim % num_heads
    q_local = ids // (head_dim * num_heads) % chunk_len
    b = ids // (head_dim * num_heads * chunk_len)
    q = q_local + start_query
    row = (b * num_heads + h) * chunk_len + q_local
    valid = ids < num_elements
    r = tl.load(residual + row * head_dim + d, valid, other=0)
    e = tl.load(exact_output + b * exact_stride_b + q * exact_stride_q + h * exact_stride_h + d * exact_stride_d, valid, other=0).to(tl.float32)
    weight = tl.load(exact_weights + row, valid, other=0)
    value = e * weight + r
    tl.store(output + b * out_stride_b + q * out_stride_q + h * out_stride_h + d * out_stride_d, value, valid)


def merge_sol_residual(query, key_centroids, value_sums, selected, sizes,
                       exact_output, exact_lse, scale, *, chunk_size=2048):
    batch, tokens, heads, dim = query.shape
    output = torch.empty_like(query)
    keys = key_centroids.float().transpose(1, 2).transpose(-1, -2)
    values = value_sums.float().transpose(1, 2)
    log_sizes = sizes.float().log()
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        chunk = stop - start
        scores = query[:, start:stop].float().transpose(1, 2) @ keys * scale
        exact_mask = selected[:, :, start // 64 : (stop + 63) // 64].repeat_interleave(64, -2)
        scores.masked_fill_(exact_mask[:, :, start % 64 : start % 64 + chunk], -torch.inf)
        residual_lse = (scores + log_sizes).logsumexp(-1)
        exact_lse_chunk = exact_lse[:, :, start:stop]
        joint_lse = torch.logaddexp(exact_lse_chunk, residual_lse)
        residual = (scores - joint_lse.unsqueeze(-1)).exp() @ values
        exact_weights = (exact_lse_chunk - joint_lse).exp()
        elements = batch * chunk * heads * dim
        _merge_residual[(triton.cdiv(elements, 512),)](
            residual, exact_output, exact_weights, output,
            elements, chunk, start, heads, dim,
            *exact_output.stride(), *output.stride(), 512, enable_fp_fusion=False,
        )
    return output
