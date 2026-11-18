import torch
from vllm.triton_utils import tl, triton
from conservative_residual import _merge_residual


@triton.jit
def _prepare_sum(scores, selected, log_sizes, exponentials, maxima,
                 rows_count: tl.constexpr, chunk: tl.constexpr, start,
                 heads: tl.constexpr, blocks: tl.constexpr,
                 sb: tl.constexpr, sh: tl.constexpr, sq: tl.constexpr, sk: tl.constexpr,
                 scale: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr):
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    cols = tl.arange(0, bn)
    q = rows % chunk + start
    h = rows // chunk % heads
    b = rows // (chunk * heads)
    valid = (rows[:, None] < rows_count) & (cols[None, :] < blocks)
    values = tl.load(scores + rows[:, None] * blocks + cols[None, :], valid, other=0) * scale
    mask = tl.load(selected + b[:, None]*sb + h[:, None]*sh + (q[:, None]//64)*sq + cols[None, :]*sk,
                   valid, other=1)
    values = tl.where(valid & ~mask, values, -float('inf'))
    weighted = values + tl.load(log_sizes + cols, cols < blocks, other=0)[None, :]
    maximum = tl.max(weighted, 1)
    maximum = tl.where(tl.abs(maximum) == float('inf'), 0., maximum)
    exps = tl.extra.cuda.libdevice.exp(weighted - maximum[:, None])
    tl.store(scores + rows[:, None] * blocks + cols[None, :], values, valid)
    tl.store(exponentials + rows[:, None] * blocks + cols[None, :], exps, valid)
    tl.store(maxima + rows, maximum, rows < rows_count)


@triton.jit
def _probabilities(scores, joint, rows_count: tl.constexpr, blocks: tl.constexpr, bs: tl.constexpr):
    ids = tl.program_id(0) * bs + tl.arange(0, bs)
    valid = ids < rows_count * blocks
    values = tl.load(scores + ids, valid, other=0)
    lse = tl.load(joint + ids // blocks, valid, other=0)
    probs = tl.extra.cuda.libdevice.exp(values - lse)
    tl.store(scores + ids, probs, valid)


def merge_sol_residual(query, key_centroids, value_sums, selected, sizes,
                       exact_output, exact_lse, scale, *, chunk_size=2048):
    batch, tokens, heads, dim = query.shape
    blocks = sizes.numel()
    output = torch.empty_like(query)
    keys = key_centroids.float().transpose(1, 2).transpose(-1, -2)
    values = value_sums.float().transpose(1, 2)
    log_sizes = sizes.float().log()
    bn = triton.next_power_of_2(blocks)
    bm = 4 if bn <= 2048 else 1
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        chunk = stop - start
        scores = query[:, start:stop].float().transpose(1, 2) @ keys
        exps = torch.empty_like(scores)
        maxima = torch.empty((batch, heads, chunk), device=query.device, dtype=torch.float32)
        _prepare_sum[(triton.cdiv(batch * heads * chunk, bm),)](
            scores, selected, log_sizes, exps, maxima, batch * heads * chunk, chunk, start,
            heads, blocks, *selected.stride(), scale, bm, bn,
            num_warps=8 if bn >= 2048 else 4, enable_fp_fusion=False,
        )
        # Retain the PyTorch reduction tree and scalar log/logaddexp operations.
        residual_lse = exps.sum(-1).log_().add_(maxima)
        exact_lse_chunk = exact_lse[:, :, start:stop]
        joint_lse = torch.logaddexp(exact_lse_chunk, residual_lse)
        _probabilities[(triton.cdiv(scores.numel(), 1024),)](
            scores, joint_lse, batch * heads * chunk, blocks, 1024, enable_fp_fusion=False,
        )
        residual = scores @ values
        exact_weights = (exact_lse_chunk - joint_lse).exp()
        elements = batch * chunk * heads * dim
        _merge_residual[(triton.cdiv(elements, 512),)](
            residual, exact_output, exact_weights, output, elements, chunk, start, heads, dim,
            *exact_output.stride(), *output.stride(), 512, enable_fp_fusion=False,
        )
    return output
