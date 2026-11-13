"""k-means assignment (Triton).

One program handles a block of rows of one head: the row tile is loaded once,
centroids are streamed in tiles, and the running argmin is carried across the
centroid loop. d2 = ||x||^2 - 2 x.c + ||c||^2 with centroid norms precomputed
in fp32. Minimum and argmin are one int32 min-reduction over a packed key
(the low mantissa bits of the clamped distance are replaced by the cluster
index), so ties go to the lowest cluster index.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fast_assign_kernel(
    x_ptr,     # *f16 [H, N, D]
    c_ptr,     # *f16 [H, C, D]
    cn2_ptr,   # *f32 [H, C]  centroid squared norms
    out_ptr,   # *i32 [H, N]
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    IBITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    offs_d = tl.arange(0, D)

    x_base = x_ptr + h.to(tl.int64) * N * D
    x = tl.load(x_base + rows.to(tl.int64)[:, None] * D + offs_d[None, :],
                mask=rmask[:, None], other=0.0)              # [BN, D] f16
    x32 = x.to(tl.float32)
    xsq = tl.sum(x32 * x32, axis=1)                          # [BN] f32

    # running elementwise-min 2D accumulator: no reduction (shuffles) inside
    # the C loop, a single tl.min at the end
    best2d = tl.full((BLOCK_N, BLOCK_C), 0x7FFFFFFF, tl.int32)
    keymask: tl.constexpr = -(1 << IBITS)                    # drop low bits

    CTAIL: tl.constexpr = (C % BLOCK_C) != 0     # partial last C tile
    c_base = c_ptr + h.to(tl.int64) * C * D
    for k0 in range(0, C, BLOCK_C):
        offs_c = k0 + tl.arange(0, BLOCK_C)
        if CTAIL:
            cm = offs_c < C
            ct = tl.load(c_base + offs_c[:, None] * D + offs_d[None, :],
                         mask=cm[:, None], other=0.0)
            cn2 = tl.load(cn2_ptr + h * C + offs_c, mask=cm, other=0.0)
        else:
            ct = tl.load(c_base + offs_c[:, None] * D + offs_d[None, :])
            cn2 = tl.load(cn2_ptr + h * C + offs_c)                    # f32
        cross = tl.dot(x, tl.trans(ct))                    # [BN, BC] f32 acc
        d2 = xsq[:, None] + (cn2[None, :] - 2.0 * cross)
        # d2 >= 0 up to rounding -> fp32 bits monotone as signed i32 (a
        # rounding-negative d2 means true distance ~0: winning the min is
        # correct); pack the global cluster index into the low IBITS bits
        key = (d2.to(tl.int32, bitcast=True) & keymask) | offs_c[None, :]
        if CTAIL:
            key = tl.where(cm[None, :], key, 0x7FFFFFFF)
        best2d = tl.minimum(best2d, key)

    best = tl.min(best2d, axis=1)
    tl.store(out_ptr + h.to(tl.int64) * N + rows,
             best & ((1 << IBITS) - 1), mask=rmask)


# Block sizes tuned for specific (H, N, C) shapes.
TUNE = {
    (14, 36864, 288): dict(BLOCK_N=128, BLOCK_C=32, num_warps=4),
    (14, 36864, 2048): dict(BLOCK_N=128, BLOCK_C=64, num_warps=4),
}


def fast_assign(x, centroids, out=None, *, BLOCK_N=None, BLOCK_C=None,
                num_warps=None, num_stages=3):
    """x [H,N,D] f16 contiguous, centroids [H,C,D] f16 contiguous.

    Returns labels [H,N] int32 (nearest centroid, lowest index on ties).
    Block params default from TUNE for known shapes, else 128/64/4."""
    tune = TUNE.get((x.shape[0], x.shape[1], centroids.shape[1]), {})
    BLOCK_N = BLOCK_N or tune.get("BLOCK_N", 128)
    BLOCK_C = BLOCK_C or tune.get("BLOCK_C", 64)
    num_warps = num_warps or tune.get("num_warps", 4)
    assert x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()
    assert centroids.is_cuda and centroids.dtype == torch.float16 \
        and centroids.is_contiguous()
    H, N, D = x.shape
    C = centroids.shape[1]
    assert centroids.shape == (H, C, D)
    ibits = max(1, (C - 1).bit_length())   # index bits packed into the key
    assert ibits <= 12, "C too large for packed int32 key"
    cn2 = (centroids.float() * centroids.float()).sum(-1)   # [H,C] f32, tiny
    if out is None:
        out = torch.empty(H, N, device=x.device, dtype=torch.int32)
    grid = (triton.cdiv(N, BLOCK_N), H)
    _fast_assign_kernel[grid](
        x, centroids, cn2, out, N, C=C, D=D, IBITS=ibits,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
