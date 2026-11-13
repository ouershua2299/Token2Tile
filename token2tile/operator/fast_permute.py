"""Per-head row gather for [H, N, D] fp16 tensors (Triton).

Rows are copied as int64 words; the int64 view is a raw bitcast."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_perm_kernel(
    x_ptr,     # *i64 [H, N, DW]
    perm_ptr,  # *i32 [H, N]
    o_ptr,     # *i64 [H, N, DW]
    N,
    DW: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GATHER: tl.constexpr,   # 1: o[i] = x[perm[i]];  0: o[perm[i]] = x[i]
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < N
    hN = h.to(tl.int64) * N
    idx = tl.load(perm_ptr + hN + rows, mask=rmask, other=0).to(tl.int64)

    offs = tl.arange(0, DW)
    po = (hN + idx)[:, None] * DW + offs[None, :]
    co = (hN + rows.to(tl.int64))[:, None] * DW + offs[None, :]
    m2 = rmask[:, None]
    if GATHER:
        tl.store(o_ptr + co, tl.load(x_ptr + po, mask=m2), mask=m2)
    else:
        tl.store(o_ptr + po, tl.load(x_ptr + co, mask=m2), mask=m2)


def _perm_copy(x, perm, out, gather, BLOCK_R, num_warps):
    assert x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()
    H, N, D = x.shape
    assert (D * 2) % 8 == 0
    DW = D * 2 // 8
    if out is None:
        out = torch.empty_like(x)
    perm_i = perm.to(torch.int32)
    if not perm_i.is_contiguous():
        perm_i = perm_i.contiguous()
    grid = (triton.cdiv(N, BLOCK_R), H)
    _copy_perm_kernel[grid](
        x.view(torch.int64), perm_i, out.view(torch.int64),
        N, DW=DW, BLOCK_R=BLOCK_R, GATHER=gather, num_warps=num_warps,
    )
    return out


def fast_gather(x, perm, out=None, *, BLOCK_R=32, num_warps=4):
    """x [H,N,D] f16, perm [H,N] int.  Returns o with o[h,i] = x[h,perm[h,i]]."""
    return _perm_copy(x, perm, out, True, BLOCK_R, num_warps)
