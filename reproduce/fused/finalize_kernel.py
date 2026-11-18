from vllm.triton_utils import tl, triton


@triton.jit
def _finish_lse(sums, maxima, exact_lse, joint, weights,
                rows_count: tl.constexpr, chunk: tl.constexpr, start,
                heads: tl.constexpr, sb: tl.constexpr, sh: tl.constexpr, sq: tl.constexpr,
                bs: tl.constexpr):
    ids = tl.program_id(0) * bs + tl.arange(0, bs)
    valid = ids < rows_count
    q = ids % chunk + start
    h = ids // chunk % heads
    b = ids // (chunk * heads)
    x = tl.extra.cuda.libdevice.log(tl.load(sums + ids, valid, other=1)) + tl.load(maxima + ids, valid, other=0)
    y = tl.load(exact_lse + b*sb + h*sh + q*sq, valid, other=0)
    maximum = tl.maximum(x, y)
    merged = maximum + tl.extra.cuda.libdevice.log1p(tl.extra.cuda.libdevice.exp(-tl.abs(x-y)))
    merged = tl.where((x == y) & (tl.abs(x) == float('inf')), x, merged)
    tl.store(joint + ids, merged, valid)
    tl.store(weights + ids, tl.extra.cuda.libdevice.exp(y - merged), valid)
