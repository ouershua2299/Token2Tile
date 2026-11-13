"""Token2Tile attention operator: one sparse attention call of one layer.

`MaskedStepDeploy.step(q, k, v, state)` runs, for q, k, v of shape [H, N, 128] (fp16):

  1. projection        Q -> Q Uq, K -> K Uk into the rank-r space              cuBLAS bmm
  2. clustering        one warm-started k-means update of the key clusters
                       (C_K = 2048) and two of the query clusters (C_Q),
                       with the per-cluster value sums                         kernels/kmeans_maintain
  3. scoring           every query against every key centroid                  cuBLAS bmm
  4. selection         per query, the shortest prefix of whole key clusters
                       that meets the token budget; the selections of one
                       query cluster are united into a packed bitmap           kernels/select_union
  5. attention         exact attention of every query cluster over its union,
                       the per-query centroid residual for the other
                       clusters, and the merge under one softmax               kernels/tile_attn

The output is in the original row order. Steps 2-4 produce the *structure* of a
call (labels, permutations, cluster sizes, centroids, union). With
`mask_refresh=R` the structure is rebuilt every R annotated steps and reused in
between; a reuse call still projects its own queries, recomputes the value sums
from its own values and runs step 5 on its own Q, K, V. `set_cfg_call` annotates
the next call with (branch, layer, step); the unconditional branch of a step
reuses the structure of the conditional one.

Two CUDA streams are used: the selection bitmap is finalised on a side stream
while keys are gathered on the main stream. The caller's stream is ordered after
both before `step` returns; there is no host synchronisation.

Fixed by the kernels: head dimension 128, C_K = 2048, non-causal attention, SM90.
"""
import hashlib
import math
import os
import sys

import torch

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from fast_assign import fast_assign                       # noqa: E402
from fast_permute import fast_gather                      # noqa: E402
from fused_update_vstats import fused_update_vstats       # noqa: E402

_KERNELS = os.path.join(CODE, "kernels")
_TAG = hashlib.sha256(CODE.encode()).hexdigest()[:8]
_CUDA_FLAGS = ["-O3", "--use_fast_math", "-std=c++17", "-lineinfo",
               "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"]


def _load(name, sources, extra_flags=(), verbose=False):
    from torch.utils.cpp_extension import load
    src = os.path.join(_KERNELS, name, "src")
    return load(name=f"t2t_{_TAG}_{name}",
                sources=[os.path.join(src, s) for s in sources],
                extra_cuda_cflags=_CUDA_FLAGS + list(extra_flags),
                extra_ldflags=["-lcuda"], verbose=verbose)


def load_select_union_ext(verbose=False):
    """Per-query selection and query-cluster union.

    run_packed(gl, counts, qperm, tile_sizes, ns, partial, unip, rowcnt, off)
    writes the union as LSB-first packed words unip [H, CQ, C/32] and the
    number of selected keys per query rowcnt [H, N]. run_split_packed first
    splits large query clusters into sub-tiles of at most split_cap rows and
    ORs their unions back, which gives the same bitmap."""
    return _load("select_union", ["main.cpp", "kernel.cu"], verbose=verbose)


def load_tile_attn_ext(verbose=False):
    """Exact attention over the unions, per-query residual, merge.

    run(q, kp, vp, q_cluster_size i32 [H,CQ], k_cluster_size i32 [H,C],
    union_bits u8 [H,CQ,C/8], QU f16 [H,N,32], cks f16 [H,C,32] (scale
    folded in), sumv f16 [H,C,128] (divided by lam_s), lam_s f32 [H],
    qlab i32 [H,N], qperm i32 [H,N] (sorted position -> original row),
    tailw 0-dim f32, out f16 [H,N,128]). q is in the original order, kp/vp
    are grouped by key cluster, out is in the original order. run_planned
    additionally takes a task workspace owned by the structure snapshot and
    a rebuild flag, so reuse calls skip task planning."""
    return _load("tile_attn", ["main.cpp", "kernel.cu"], verbose=verbose)


def load_kmm_ext(verbose=False):
    """Warm-started k-means maintenance for queries and keys in one call.

    run(QU, KU, V, cq0, ck0 -> qlab, cq2, klab, ck2, qcounts, qperm, kcounts,
    kperm, V_sorted, sumv, lam_s; pin_tail). Queries take two assignment
    passes, keys one; empty clusters keep their centroid; ties go to the
    lowest cluster index. sumv holds the value sum of every key cluster
    divided by the per-head lam_s = max(1, max|sum| / 32768) so that it fits
    fp16. pin_tail reserves the last key cluster for that many trailing keys.
    run_vstats_gather(V, kperm, counts -> V_sorted, sumv, lam_s) recomputes
    only the value statistics for a reuse call. Deterministic."""
    return _load("kmeans_maintain", ["main.cpp", "kernels.cu"],
                 extra_flags=["-DANGR=6", "-DAMINB=2", "-DAW_NTILE=256"],
                 verbose=verbose)


# ---------------------------------------------------------------------------
# low-rank space
# ---------------------------------------------------------------------------

def _eigh_robust(M):
    """cuSOLVER occasionally fails to converge on these small fp64 matrices;
    fall back to LAPACK on the CPU."""
    try:
        return torch.linalg.eigh(M)
    except torch._C._LinAlgError:
        e, V = torch.linalg.eigh(M.cpu())
        return e.to(M.device), V.to(M.device)


def _svd_robust(M):
    try:
        return torch.linalg.svd(M)
    except torch._C._LinAlgError:
        U, S, Vh = torch.linalg.svd(M.cpu())
        return U.to(M.device), S.to(M.device), Vh.to(M.device)


def online_u(qf, kf, r=32):
    """Matched rank-r projections Uq, Uk [H, D, r] (fp32) that minimise
    ||Q K^T - (Q Uq)(K Uk)^T||_F: with the second moments Mq, Mk and the SVD
    Mq^(1/2) Mk^(1/2) = U S V^T, Uq = Mq^(-1/2) U_r S_r^(1/2) and
    Uk = Mk^(-1/2) V_r S_r^(1/2)."""
    for name, t in (("q", qf), ("k", kf)):
        bad = ~torch.isfinite(t)
        if bool(bad.any()):
            raise RuntimeError(
                f"online_u: non-finite {name} input; per-head counts "
                f"{bad.sum(dim=(1, 2)).tolist()}, shape {tuple(t.shape)}")
    Uqs, Uks = [], []
    for h in range(qf.shape[0]):
        q, k = qf[h].double(), kf[h].double()
        Mq = (q.T @ q) / q.shape[0]
        Mk = (k.T @ k) / k.shape[0]
        eq, Vq = _eigh_robust(Mq)
        ek, Vk = _eigh_robust(Mk)
        eq = eq.clamp_min(1e-8); ek = ek.clamp_min(1e-8)
        Rq = Vq @ torch.diag(eq.sqrt()) @ Vq.T
        Rk = Vk @ torch.diag(ek.sqrt()) @ Vk.T
        Rqi = Vq @ torch.diag(eq.rsqrt()) @ Vq.T
        Rki = Vk @ torch.diag(ek.rsqrt()) @ Vk.T
        U, S, Vh = _svd_robust(Rq.T @ Rk)
        sr = S[:r].sqrt()
        Uqs.append((Rqi.T @ U[:, :r] * sr).float())
        Uks.append((Rki @ Vh[:r].T * sr).float())
    return torch.stack(Uqs), torch.stack(Uks)


class MaskedStepDeploy:
    """Persistent buffers and kernels for one (H, N, C_Q) attention shape.

    s_frac: per-query token budget as a fraction of N (`ns_t` holds the token
        count and may be overwritten between calls).
    mask_refresh: rebuild the structure every this many annotated steps.
    cfg_reuse: enable structure reuse across CFG branches even when
        mask_refresh is 1.
    pin_tail: the last pin_tail keys form one reserved key cluster that every
        query executes exactly (H3 text and audio tokens).
    split_cap: bound on the rows one selection block processes; "auto" splits
        query clusters at the mean cluster size when the grid is shallow."""

    def __init__(self, H, N, CQ, *, C=2048, D=128, r=32, s_frac=0.05,
                 tailw=1.0, device="cuda", mask_refresh=1, cfg_reuse=False,
                 pin_tail=0, split_cap="auto"):
        if C != 2048 or D != 128:
            raise ValueError("the kernels require C=2048 and D=128")
        self.H, self.N, self.CQ, self.C, self.D, self.r = H, N, CQ, C, D, r
        self.ns = max(1, int(s_frac * N))
        self.tailw = tailw
        self.dev = dev = torch.device(device)
        self.scale = 1.0 / math.sqrt(D)
        self.pin_tail = int(pin_tail)
        self.ext = load_select_union_ext()
        self.tile_attn_ext = load_tile_attn_ext()
        self.kmm_ext = load_kmm_ext()
        if split_cap == "auto":
            # Splitting pays only when the selection grid is shallow (few
            # waves of blocks), where the largest query cluster dominates.
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            waves = (H * CQ) / (8.0 * sms)
            split_cap = (N + CQ - 1) // CQ if waves < 6 else 0
        self.split_cap = int(split_cap) if split_cap else 0

        f16, f32, i32 = torch.float16, torch.float32, torch.int32
        # persistent buffers, allocated once
        self.QU = torch.empty(H, N, r, device=dev, dtype=f16)
        self.KU = torch.empty(H, N, r, device=dev, dtype=f16)
        self.gl = torch.empty(H, N, C, device=dev, dtype=f16)
        self.klab = torch.empty(H, N, device=dev, dtype=i32)
        self.qlab = torch.empty(H, N, device=dev, dtype=i32)
        self.rowcnt = torch.empty(H, N, device=dev, dtype=i32)
        self.kp = torch.empty(H, N, D, device=dev, dtype=f16)
        self.vp = torch.empty(H, N, D, device=dev, dtype=f16)
        self.tile_attn_out = torch.empty(H, N, D, device=dev, dtype=f16)
        qt = 128 if N >= 8192 else 64
        max_tasks = H * (CQ + (N + qt - 1) // qt + 1)
        task_offset = (H * (C + 1) + 7) & ~3
        self._tile_attn_plan_ints = task_offset + 4 * max_tasks + H * CQ
        self.tailw_t = torch.tensor(float(tailw), device=dev, dtype=f32)
        self._kmm_cq2 = torch.empty(H, CQ, r, device=dev, dtype=f16)
        self._kmm_ck2 = torch.empty(H, C, r, device=dev, dtype=f32)
        self._kmm_kcounts = torch.empty(H, C, device=dev, dtype=i32)
        self._kmm_sumv = torch.empty(H, C, D, device=dev, dtype=f16)
        self._kmm_lam = torch.empty(H, device=dev, dtype=f32)
        self.ns_t = torch.tensor(self.ns, device=dev, dtype=i32)

        # Structure reuse. A full call (branch 0 and snapshot age >=
        # mask_refresh) rebuilds the structure and stores it per key (layer);
        # every other annotated call reuses the stored snapshot. Calls that
        # are not annotated through set_cfg_call are always full.
        self.cfg_reuse = bool(cfg_reuse)
        self.mask_refresh = max(1, int(mask_refresh))
        self._cfg_on = self.cfg_reuse or self.mask_refresh > 1
        self._cfg_hold = {}          # key (layer) -> structure snapshot
        self._cfg_parity = None      # annotation of the next call
        self._cfg_key = 0
        self._cfg_step = 0
        self._cfg_snap_q = None      # query half of the snapshot in flight

        selector_tiles = CQ
        if self.split_cap:
            cap = self.split_cap
            # worst case: every cluster has one partial sub-tile plus full ones
            selector_tiles = CQ + (N + cap - 1) // cap
            self._tsv = torch.empty(H, selector_tiles, device=dev, dtype=i32)
            self._qoff_v = torch.empty(H, selector_tiles, device=dev, dtype=i32)
            self.vbase32 = torch.empty(H, CQ, device=dev, dtype=i32)
            self.nsplit32 = torch.empty(H, CQ, device=dev, dtype=i32)
        self._selector_partial = torch.empty(
            H, selector_tiles, 6 * 128, device=dev, dtype=torch.int16)
        self._selector_off = torch.empty(H, CQ, device=dev, dtype=i32)

        self.s_main = torch.cuda.Stream(device=dev)
        self.s_side = torch.cuda.Stream(device=dev)
        self.evt_qu = torch.cuda.Event()
        self.evt_qlab = torch.cuda.Event()
        self.evt_uni = torch.cuda.Event()
        self.evt_unip = torch.cuda.Event()
        self.evt_kmm = torch.cuda.Event()

    # -- state init ---------------------------------------------------------
    def init_state(self, q, k, *, warm_iters=4, seed=0):
        """Projections and initial centroids of one layer. The projections are
        kept for the whole request; the centroids warm-start every rebuild."""
        N, r = self.N, self.r
        Uq, Uk = online_u(q.float(), k.float(), r)
        Uq = Uq.to(torch.float16).contiguous()
        Uk = Uk.to(torch.float16).contiguous()
        KU = torch.bmm(k, Uk).contiguous()
        QU = torch.bmm(q, Uq).contiguous()
        g = torch.Generator(device="cpu").manual_seed(seed)
        ck = KU[:, torch.randperm(N, generator=g)[: self.C].sort().values,
                :].contiguous()
        cq = QU[:, torch.randperm(N, generator=g)[: self.CQ].sort().values,
                :].contiguous()
        for _ in range(warm_iters):
            lab = fast_assign(KU, ck)
            ck, _, _ = fused_update_vstats(KU, lab, ck, None)
            lab = fast_assign(QU, cq)
            cq, _, _ = fused_update_vstats(QU, lab, cq, None)
        torch.cuda.synchronize()
        return {"Uq": Uq, "Uk": Uk, "ck": ck.contiguous(),
                "cq": cq.contiguous()}

    def set_cfg_call(self, parity, key=0, step=0):
        """Annotate the next step() call (drives reuse across steps and CFG branches).

        parity: CFG branch within the denoising step (0 = the branch allowed
        to rebuild). key: snapshot slot, one per layer. step: denoising step.
        A branch-0 call rebuilds when step - snapshot_step >= mask_refresh,
        so the first sparse call of a layer is always full."""
        self._cfg_parity = int(parity)
        self._cfg_key = key
        self._cfg_step = int(step)

    def reset_cfg_reuse(self):
        """Drop all snapshots and any pending annotation (between requests)."""
        self._cfg_hold.clear()
        self._cfg_parity = None
        self._cfg_snap_q = None

    # -- the attention call --------------------------------------------------
    def step(self, q, k, v, state):
        """q, k, v: [H,N,128] f16 contiguous, original row order.
        state: dict(Uq, Uk, ck, cq) f16; ck/cq are replaced by the centroids
        that warm-start the next call. Returns out [H,N,128] f16."""
        # Inputs and state may still be produced on the caller's stream.
        caller_stream = torch.cuda.current_stream(q.device)
        if caller_stream != self.s_main:
            self.s_main.wait_stream(caller_stream)
        H, N, C, CQ = self.H, self.N, self.C, self.CQ
        s0, s1 = self.s_main, self.s_side
        # snap != None -> reuse call
        snap = None
        held = None
        cfg_parity = self._cfg_parity
        self._cfg_parity = None      # an unannotated call never consumes a stale annotation
        if self._cfg_on and cfg_parity is not None:
            held = self._cfg_hold.get(self._cfg_key)
            age = None if held is None else self._cfg_step - held["step"]
            # age < 0: the step counter was reset (new request) -> full call
            refresh_due = (age is None or age < 0
                           or age >= self.mask_refresh)
            if held is not None and age >= 0 \
                    and not (cfg_parity == 0 and refresh_due):
                snap = held
        # A rebuild overwrites the tensors owned by the snapshot it replaces;
        # the first rebuild of a key allocates them.
        recycle_snapshot = held is not None and snap is None
        qlab_work = held["qlab"] if recycle_snapshot else self.qlab
        klab_work = held["klab"] if recycle_snapshot else self.klab
        tile_attn_plan = snap["tile_attn_plan"] if snap is not None else None
        selector_unip = None

        with torch.cuda.stream(s0):
            # 1. projections. A reuse call projects its queries with the
            # snapshot's Uq so that QU and the stored centroids share a basis.
            torch.bmm(q, snap["Uq"] if snap is not None else state["Uq"],
                      out=self.QU)
            if snap is None:
                torch.bmm(k, state["Uk"], out=self.KU)
            else:
                fast_gather(k, snap["kperm"], out=self.kp)
            self.evt_qu.record(s0)
            if snap is None:
                # 2. clustering
                if recycle_snapshot:
                    kmm_qcounts = held["tile_sizes"]
                    kmm_qperm = held["qperm"]
                    kmm_kperm = held["kperm"]
                    kmm_kcounts = held["counts"]
                else:
                    kmm_qcounts = torch.empty(
                        H, CQ, device=self.dev, dtype=torch.int32)
                    kmm_qperm = torch.empty(
                        H, N, device=self.dev, dtype=torch.int32)
                    kmm_kperm = torch.empty(
                        H, N, device=self.dev, dtype=torch.int32)
                    kmm_kcounts = self._kmm_kcounts
                self.kmm_ext.run(self.QU, self.KU, v,
                                 state["cq"], state["ck"],
                                 qlab_work, self._kmm_cq2, klab_work,
                                 self._kmm_ck2, kmm_qcounts, kmm_qperm,
                                 kmm_kcounts, kmm_kperm, self.vp,
                                 self._kmm_sumv, self._kmm_lam,
                                 self.pin_tail)
                self.evt_kmm.record(s0)

        with torch.cuda.stream(s1):
            s1.wait_event(self.evt_qu)
            if snap is not None:
                tile_sizes = snap["tile_sizes"]
                qperm = snap["qperm"]
            else:
                s1.wait_event(self.evt_kmm)
                tile_sizes = kmm_qcounts
                qperm = kmm_qperm
                if self._cfg_on:
                    # Query half of the snapshot. On the first rebuild qlab
                    # is the shared buffer and is cloned; later rebuilds
                    # write the old snapshot's tensor directly.
                    self._cfg_snap_q = {
                        "qlab": (qlab_work if recycle_snapshot
                                 else qlab_work.clone()),
                        "qperm": qperm,
                        "tile_sizes": tile_sizes,
                    }
                self.evt_qlab.record(s1)

        with torch.cuda.stream(s0):
            if snap is not None:
                # Reuse: only the value statistics are recomputed, from this
                # call's V through the stored key grouping. The centroids are
                # not advanced.
                counts = snap["counts"]
                self.kmm_ext.run_vstats_gather(
                    v, snap["kperm"], counts, self.vp,
                    self._kmm_sumv, self._kmm_lam)
                kperm = snap["kperm"]
                cks = snap["cks"]
            else:
                counts = kmm_kcounts
                ck_new = self._kmm_ck2
                # fresh tensors, so the next call's inputs never alias the
                # kernel's output buffers
                state["ck"] = ck_new.to(torch.float16).contiguous()
                state["cq"] = self._kmm_cq2.clone()
                kperm = kmm_kperm
                # 3. scoring (scale folded into the centroids)
                cks = (ck_new * self.scale).to(torch.float16).contiguous()
                torch.bmm(self.QU, cks.transpose(1, 2), out=self.gl)
                # 4. selection and union
                s0.wait_event(self.evt_qlab)
                selector_unip = (held["unip"] if recycle_snapshot else
                                 torch.empty(H, CQ, C // 32, device=self.dev,
                                             dtype=torch.int32))
                if self.split_cap:
                    self.ext.run_split_packed(
                        self.gl, counts, qperm, tile_sizes,
                        self.split_cap, self.ns_t,
                        self._selector_partial, selector_unip,
                        self.rowcnt, self._tsv, self._qoff_v,
                        self.vbase32, self.nsplit32)
                else:
                    self.ext.run_packed(
                        self.gl, counts, qperm, tile_sizes, self.ns_t,
                        self._selector_partial, selector_unip,
                        self.rowcnt, self._selector_off)
                if self._cfg_on:
                    # Publish the snapshot (never written between rebuilds).
                    tile_attn_plan = (held["tile_attn_plan"]
                                      if recycle_snapshot else torch.empty(
                                          self._tile_attn_plan_ints,
                                          device=self.dev, dtype=torch.int32))
                    self._cfg_hold[self._cfg_key] = {
                        **self._cfg_snap_q,
                        "klab": (klab_work if recycle_snapshot
                                 else klab_work.clone()),
                        "kperm": kperm,
                        "counts": (counts if recycle_snapshot
                                   else counts.clone()),
                        "cks": cks,
                        "Uq": state["Uq"],
                        "unip": None,      # filled on the side stream below
                        "tile_attn_plan": tile_attn_plan,
                        "step": self._cfg_step,
                    }
            sumv = self._kmm_sumv
            lam_s = self._kmm_lam
            self.evt_uni.record(s0)

        with torch.cuda.stream(s1):
            s1.wait_event(self.evt_uni)
            # reuse calls must see the snapshot's labels: self.qlab is a
            # shared buffer that another layer may have overwritten
            qlab_cur = snap["qlab"] if snap is not None else qlab_work
            if snap is not None:
                unip = snap["unip"]
            else:
                unip = selector_unip
                if self.pin_tail:
                    # every query cluster executes the reserved cluster
                    unip.view(torch.uint8).reshape(
                        unip.shape[0], unip.shape[1], -1
                    )[:, :, (self.C - 1) >> 3] |= 1 << ((self.C - 1) & 7)
                if self._cfg_on:
                    self._cfg_hold[self._cfg_key]["unip"] = unip
            self.evt_unip.record(s1)

        with torch.cuda.stream(s0):
            if snap is None:
                # V was grouped by the clustering kernel; group K the same way
                fast_gather(k, kperm, out=self.kp)
            # 5. attention over the unions + residual + merge, one call
            s0.wait_event(self.evt_unip)
            union_bits = unip.view(torch.uint8).reshape(H, CQ, C // 8)
            if self._cfg_on:
                self.tile_attn_ext.run_planned(
                    q, self.kp, self.vp, tile_sizes, counts, union_bits,
                    self.QU, cks, sumv, lam_s, qlab_cur, qperm,
                    self.tailw_t, self.tile_attn_out, tile_attn_plan,
                    snap is None)
            else:
                self.tile_attn_ext.run(
                    q, self.kp, self.vp, tile_sizes, counts, union_bits,
                    self.QU, cks, sumv, lam_s, qlab_cur, qperm,
                    self.tailw_t, self.tile_attn_out)
            out = self.tile_attn_out

        # order the caller's stream after ours instead of a host sync
        cur = torch.cuda.current_stream()
        cur.wait_stream(s0)
        cur.wait_stream(s1)
        return out
