"""k-means centroid update and per-cluster value statistics in one pass (Triton).

Labels are sorted once; runs of identical cluster ids are reduced with one
atomic add per run, following the segmented centroid update of SVG2. Key
features and values share the pass, giving new centroids [H,C,DK], counts
[H,C] and mean values [H,C,DV]. Accumulation is fp32. v=None skips the value
statistics.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_update_kernel(
    ku_ptr,       # *f16 [H, N, DK]  original order
    v_ptr,        # *f16 [H, N, DV]  original order (unused if !HAS_V)
    sidx_ptr,     # *i32 [H, N]  row indices after label sort
    scid_ptr,     # *i32 [H, N]  cluster ids in sorted order
    sumk_ptr,     # *f32 [H, C, DK]  (zeroed)
    sumv_ptr,     # *f32 [H, C, DV]  (zeroed, unused if !HAS_V)
    cnt_ptr,      # *i32 [H, C]      (zeroed)
    N,
    C,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_V: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    start = pid * BLOCK_N
    if start >= N:
        return

    offs = start + tl.arange(0, BLOCK_N)
    tmask = offs < N
    hN = h.to(tl.int64) * N
    ids = tl.load(scid_ptr + hN + offs, mask=tmask, other=-1)      # [BN]
    ridx = tl.load(sidx_ptr + hN + offs, mask=tmask, other=0)      # [BN]
    ridx64 = ridx.to(tl.int64)

    first_id = tl.load(scid_ptr + hN + start)
    last_pos = tl.minimum(start + BLOCK_N, N) - 1
    last_id = tl.load(scid_ptr + hN + last_pos)

    offs_dk = tl.arange(0, DK)
    ku_base = ku_ptr + h.to(tl.int64) * N * DK
    ku_off = ridx64[:, None] * DK + offs_dk[None, :]
    if HAS_V:
        offs_dv = tl.arange(0, DV)
        v_base = v_ptr + h.to(tl.int64) * N * DV
        v_off = ridx64[:, None] * DV + offs_dv[None, :]

    hC = h.to(tl.int64) * C
    for cid in range(first_id, last_id + 1):
        cmask = ids == cid
        csize = tl.sum(cmask.to(tl.int32))
        if csize != 0:
            ku_rows = tl.load(ku_base + ku_off, mask=cmask[:, None],
                              other=0.0).to(tl.float32)             # [BN,DK]
            tl.atomic_add(sumk_ptr + (hC + cid) * DK + offs_dk,
                          tl.sum(ku_rows, axis=0))
            if HAS_V:
                v_rows = tl.load(v_base + v_off, mask=cmask[:, None],
                                 other=0.0).to(tl.float32)          # [BN,DV]
                tl.atomic_add(sumv_ptr + (hC + cid) * DV + offs_dv,
                              tl.sum(v_rows, axis=0))
            tl.atomic_add(cnt_ptr + hC + cid, csize)


def fused_update_vstats(ku, labels, old_centroids, v=None, *, BLOCK_N=32,
                        num_warps=2, return_fp32=False):
    """One sorted-segmented pass producing centroids, counts and V sums.

    ku [H,N,DK] f16, labels [H,N] int32/int64, old_centroids [H,C,DK] f16
    (fills empty clusters), v [H,N,DV] f16 or None.

    Returns (new_centroids [H,C,DK], counts [H,C] i32, mean_v [H,C,DV]|None).
    Outputs are cast to ku/v dtype unless return_fp32."""
    assert ku.is_cuda and ku.is_contiguous()
    H, N, DK = ku.shape
    C = old_centroids.shape[1]
    if v is not None:
        assert v.is_contiguous() and v.shape[:2] == (H, N)
        DV = v.shape[2]
    else:
        DV = DK  # dummy constexpr; branch compiled out

    sorted_cid, sorted_idx = torch.sort(labels.to(torch.int32), dim=-1)
    sorted_idx = sorted_idx.to(torch.int32)

    sums_k = torch.zeros(H, C, DK, device=ku.device, dtype=torch.float32)
    sums_v = (torch.zeros(H, C, DV, device=ku.device, dtype=torch.float32)
              if v is not None else sums_k)  # dummy ptr when unused
    counts = torch.zeros(H, C, device=ku.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), H)
    _fused_update_kernel[grid](
        ku, v if v is not None else ku, sorted_idx, sorted_cid,
        sums_k, sums_v, counts,
        N, C, DK=DK, DV=DV, BLOCK_N=BLOCK_N, HAS_V=v is not None,
        num_warps=num_warps,
    )

    cnt_f = counts.to(torch.float32).clamp_min(1.0)[..., None]
    newc = sums_k / cnt_f
    empty = (counts == 0)[..., None]
    newc = torch.where(empty, old_centroids.to(torch.float32), newc)
    mean_v = sums_v / cnt_f if v is not None else None
    if not return_fp32:
        newc = newc.to(ku.dtype)
        if mean_v is not None:
            mean_v = mean_v.to(v.dtype)
    return newc, counts, mean_v
