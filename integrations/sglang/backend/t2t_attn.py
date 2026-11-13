"""Token2Tile sparse-attention backend for Wan2.1 and MiniMax-H3 on Hopper (SM90a).

The backend supplies the SGLang lifecycle around the Token2Tile operator
(``token2tile/operator``): a dense FlashAttention prefix, per-request low-rank
projections and cluster state, and the per-(step window, layer) token budgets.

Wan2.1 goes through ``forward`` (one call per CFG branch); MiniMax-H3 goes
through ``forward_varlen`` on its packed [text | audio | video] sequence:
context rows are computed densely, context keys stay exact for every video
query, and the video rows run through the operator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

logger = init_logger(__name__)

def _ulysses_world_size() -> int:
    """Ulysses (head-sharding) degree of this rank's SP group; 1 when the
    process group is not initialised (single-process tests)."""
    try:
        return max(1, int(get_ulysses_parallel_world_size()))
    except Exception:  # noqa: BLE001 - distributed state not initialised
        return 1


_MODULE_LOCK = threading.Lock()
_OPERATOR_MODULES: dict[tuple[str, str], ModuleType] = {}
_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[tuple[Any, ...], "_T2TRuntime"] = {}


@dataclass(frozen=True)
class T2TProfile:
    """Frozen budget policy for one Wan workload geometry."""

    name: str
    num_heads: int
    seq_len: int
    total_steps: int
    ns_base: int
    dense_prefix_steps: int
    cq: int
    ck: int
    step_profile: tuple[float, float, float, float]
    layer_profile: tuple[float, ...]
    cfg_mult: tuple[float, float] = (1.0, 1.0)


_WAN_FROZEN_STEP_PROFILE = (1.574482, 0.985902, 0.555188, 0.884428)
_WAN_FROZEN_LAYER_PROFILE = (
    1.189098,
    1.123097,
    1.278643,
    0.915148,
    0.956427,
    0.537588,
)

# These are the final lossless Phase-C policies, not the earlier error-probe
# profile.  Auto selection is deliberately exact and fails closed for unknown
# sequence lengths or denoising schedules.
_WAN_PROFILES = {
    "wan21_t2v_n32760_t50_lossless": T2TProfile(
        name="wan21_t2v_n32760_t50_lossless",
        num_heads=12,
        seq_len=32760,
        total_steps=50,
        ns_base=1050,
        dense_prefix_steps=12,
        cq=256,
        ck=2048,
        step_profile=_WAN_FROZEN_STEP_PROFILE,
        layer_profile=_WAN_FROZEN_LAYER_PROFILE,
    ),
    "wan21_t2v_1_3b_n75600_t50_lossless": T2TProfile(
        name="wan21_t2v_1_3b_n75600_t50_lossless",
        num_heads=12,
        seq_len=75600,
        total_steps=50,
        ns_base=2424,
        dense_prefix_steps=12,
        cq=512,
        ck=2048,
        step_profile=_WAN_FROZEN_STEP_PROFILE,
        layer_profile=_WAN_FROZEN_LAYER_PROFILE,
    ),
    "wan21_t2v_14b_n75600_t50_lossless": T2TProfile(
        name="wan21_t2v_14b_n75600_t50_lossless",
        num_heads=40,
        seq_len=75600,
        total_steps=50,
        ns_base=2424,
        dense_prefix_steps=12,
        cq=512,
        ck=2048,
        step_profile=_WAN_FROZEN_STEP_PROFILE,
        layer_profile=_WAN_FROZEN_LAYER_PROFILE,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batched_whitening_enabled() -> bool:
    """t2t_batched_whitening: compute the per-request whitening basis for all
    heads in one batched fp64 eigh/SVD instead of a Python loop with a cuSOLVER
    sync per head.  Same estimator, same fp64 math; on 8-way Ulysses the
    per-head loop (~15 ms/head, 50-80 states per request) dominated the
    first-sparse-step cost."""
    try:
        cfg = getattr(get_global_server_args(), "attention_backend_config", None) or {}
        return bool(cfg.get("t2t_batched_whitening", False))
    except Exception:  # noqa: BLE001 - server args not initialised (tests)
        return False


def _batched_eigh(matrix: torch.Tensor):
    """Batched symmetric eigh with the archived ridge/CPU fallbacks."""
    scale = matrix.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)[:, None, None]
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    for epsilon in (0.0, 1e-10, 1e-8, 1e-6):
        try:
            return torch.linalg.eigh(matrix + epsilon * scale * eye)
        except torch._C._LinAlgError:
            continue
    values, vectors = torch.linalg.eigh(matrix.cpu())
    return values.to(matrix.device), vectors.to(matrix.device)


def _batched_svd(matrix: torch.Tensor):
    try:
        return torch.linalg.svd(matrix)
    except torch._C._LinAlgError:
        u, singular, vh = torch.linalg.svd(matrix.cpu())
        return u.to(matrix.device), singular.to(matrix.device), vh.to(matrix.device)


def _batched_online_u(qf: torch.Tensor, kf: torch.Tensor, r: int = 32):
    """Batched-over-heads version of the archived ``online_u`` whitening:
    Gram -> fp64 eigh -> symmetric sqrt / inv-sqrt -> SVD of Pq^T Pk -> rank-r
    factors.  Identical estimator to ``_h3_online_u`` / ``online_u``; only the
    head loop is replaced by batched linear algebra."""
    for name, t in (("q", qf), ("k", kf)):
        if not bool(torch.isfinite(t).all()):
            raise RuntimeError(f"[t2t] online_u: non-finite {name} input {tuple(t.shape)}")
    q = qf.double()
    k = kf.double()
    n = q.shape[1]
    gq = torch.bmm(q.transpose(1, 2), q) / n
    gk = torch.bmm(k.transpose(1, 2), k) / n
    eq, vq = _batched_eigh(gq)
    ek, vk = _batched_eigh(gk)
    eq = eq.clamp_min(1e-8)
    ek = ek.clamp_min(1e-8)
    vqt = vq.transpose(1, 2)
    vkt = vk.transpose(1, 2)
    pq = vq @ (eq.sqrt().unsqueeze(-1) * vqt)
    pk = vk @ (ek.sqrt().unsqueeze(-1) * vkt)
    pqi = vq @ (eq.rsqrt().unsqueeze(-1) * vqt)
    pki = vk @ (ek.rsqrt().unsqueeze(-1) * vkt)
    u, singular, vh = _batched_svd(pq.transpose(1, 2) @ pk)
    root = singular[:, :r].sqrt().unsqueeze(1)  # [H,1,r]
    uq = (pqi.transpose(1, 2) @ u[:, :, :r]) * root
    uk = (pki @ vh[:, :r, :].transpose(1, 2)) * root
    return uq.float().contiguous(), uk.float().contiguous()


def _fast_online_u(qf: torch.Tensor, kf: torch.Tensor, r: int = 32):
    """t2t_fast_init: same estimator as _batched_online_u, but the N-length
    Gram products run in fp32 (tensor cores) and only the 128x128 eigh/SVD run
    in fp64; no host-synchronising finiteness check (non-finite Grams still fail
    in eigh and fall back)."""
    n = qf.shape[1]
    gq = (torch.bmm(qf.transpose(1, 2), qf) / n).double()
    gk = (torch.bmm(kf.transpose(1, 2), kf) / n).double()
    eq, vq = _batched_eigh(gq)
    ek, vk = _batched_eigh(gk)
    eq = eq.clamp_min(1e-8)
    ek = ek.clamp_min(1e-8)
    vqt = vq.transpose(1, 2)
    vkt = vk.transpose(1, 2)
    pq = vq @ (eq.sqrt().unsqueeze(-1) * vqt)
    pk = vk @ (ek.sqrt().unsqueeze(-1) * vkt)
    pqi = vq @ (eq.rsqrt().unsqueeze(-1) * vqt)
    pki = vk @ (ek.rsqrt().unsqueeze(-1) * vkt)
    # Top-r SVD of M = Pq^T Pk via fp64 eigh of M^T M (batched 128x128 SVD is
    # ~25x slower on GPU); U_r = M V_r / s_r.
    m = pq.transpose(1, 2) @ pk
    w, v = _batched_eigh(m.transpose(1, 2) @ m)
    v_r = v[:, :, -r:].flip(-1)
    s_r = w[:, -r:].flip(-1).clamp_min(1e-30).sqrt()
    u_r = (m @ v_r) / s_r.unsqueeze(1)
    root = s_r.sqrt().unsqueeze(1)
    uq = (pqi.transpose(1, 2) @ u_r) * root
    uk = (pki @ v_r) * root
    return uq.float().contiguous(), uk.float().contiguous()


def _det_centroid_update(x: torch.Tensor, labels: torch.Tensor, old: torch.Tensor):
    """Deterministic replacement for fused_update_vstats(x, labels, old, None):
    sums are accumulated as int64 fixed point (2^-20 resolution) with integer
    scatter-add, which is order-independent, so identical inputs always give
    identical centroids (fp32 atomics can differ in the last bits)."""
    H, N, D = x.shape
    C = old.shape[1]
    lab = labels.to(torch.int64)
    xi = torch.round(x.float() * 1048576.0).to(torch.int64)
    sums = torch.zeros(H, C, D, dtype=torch.int64, device=x.device).scatter_add_(
        1, lab.unsqueeze(-1).expand(H, N, D), xi)
    counts = torch.zeros(H, C, dtype=torch.int64, device=x.device).scatter_add_(
        1, lab, torch.ones_like(lab))
    cent = (sums.double() / 1048576.0) / counts.clamp_min(1).unsqueeze(-1).double()
    cent = torch.where((counts > 0).unsqueeze(-1), cent, old.double())
    return cent.to(old.dtype).contiguous(), counts.to(torch.int32), None


def _make_fast_init_state(deploy, module):
    """Replacement for MaskedStepDeploy.init_state: fp32-Gram whitening, GPU
    centroid sampling, no trailing torch.cuda.synchronize()."""

    def init_state(q, k, *, warm_iters=4, seed=0):
        H, N, r = deploy.H, deploy.N, deploy.r
        Uq, Uk = _fast_online_u(q.float(), k.float(), r)
        Uq = Uq.to(torch.float16).contiguous()
        Uk = Uk.to(torch.float16).contiguous()
        KU = torch.bmm(k, Uk).contiguous()
        QU = torch.bmm(q, Uq).contiguous()
        g = torch.Generator(device=q.device).manual_seed(seed)
        ck = KU[:, torch.randperm(N, generator=g, device=q.device)[: deploy.C].sort().values, :].contiguous()
        cq = QU[:, torch.randperm(N, generator=g, device=q.device)[: deploy.CQ].sort().values, :].contiguous()
        for _ in range(warm_iters):
            lab = module.fast_assign(KU, ck)
            ck, _, _ = _det_centroid_update(KU, lab, ck)
            lab = module.fast_assign(QU, cq)
            cq, _, _ = _det_centroid_update(QU, lab, cq)
        return {"Uq": Uq, "Uk": Uk, "ck": ck.contiguous(), "cq": cq.contiguous()}

    return init_state


def _operator_dir() -> Path:
    """Directory of the Token2Tile operator (deploy_step.py, Triton helpers,
    kernels/{tile_attn,select_union,kmeans_maintain}/src).
    ``T2T_OPERATOR_DIR`` overrides the installed ``token2tile`` package."""
    value = os.environ.get("T2T_OPERATOR_DIR", "").strip()
    if value:
        root = Path(value)
    else:
        try:
            import token2tile
        except ImportError as exc:
            raise RuntimeError(
                "Token2Tile operator not found: install the token2tile package "
                "or set T2T_OPERATOR_DIR"
            ) from exc
        root = Path(token2tile.operator_dir())
    root = root.resolve()
    for required in ("deploy_step.py", "kernels/tile_attn/src/kernel.cu",
                     "kernels/select_union/src/kernel.cu", "kernels/kmeans_maintain/src/kernels.cu"):
        if not (root / required).is_file():
            raise RuntimeError(f"Token2Tile operator is incomplete: missing {root / required}")
    return root


def _load_operator(tag: str) -> ModuleType:
    """One operator module instance per model path (``wan`` / ``h3``)."""
    root = _operator_dir()
    key = (str(root), tag)
    with _MODULE_LOCK:
        cached = _OPERATOR_MODULES.get(key)
        if cached is not None:
            return cached
        logger.info("Token2Tile %s: operator %s", tag, root)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(f"_sglang_t2t_operator_{tag}", root / "deploy_step.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the Token2Tile operator: {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if tag == "h3":
            module.online_u = _batched_online_u if _batched_whitening_enabled() else _h3_online_u
        elif _batched_whitening_enabled():
            module.online_u = _batched_online_u
        _OPERATOR_MODULES[key] = module
        return module


@dataclass
class T2TRequestCache:
    states: dict[tuple[int, int], dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    call_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    runtime_key: tuple[Any, ...] | None = None
    initialized_runtime: bool = False


@dataclass
class T2TAttentionMetadata(AttentionMetadata):
    step_index: int
    total_steps: int
    profile_name: str
    ns_base: int | None
    dense_prefix_steps: int | None
    cq: int | None
    ck: int | None
    step_profile: tuple[float, ...] | None
    layer_profile: tuple[float, ...] | None
    cfg_mult: tuple[float, ...] | None
    ns_table: str | None
    warm_iters: int
    mask_refresh: int
    cfg_reuse: bool
    use_kmm: bool
    cache: T2TRequestCache
    # Dense cross-attention receives the same forward-context metadata and
    # uses these two mutable fields exactly like FlashAttentionMetadata.
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None


class T2TAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        step_index: int,
        total_steps: int,
        cache: T2TRequestCache,
        profile_name: str = "auto",
        ns_base: int | None = None,
        dense_prefix_steps: int | None = None,
        cq: int | None = None,
        ck: int | None = None,
        step_profile: tuple[float, ...] | list[float] | None = None,
        layer_profile: tuple[float, ...] | list[float] | None = None,
        cfg_mult: tuple[float, ...] | list[float] | None = None,
        ns_table: str | None = None,
        warm_iters: int = 4,
        mask_refresh: int = 4,
        cfg_reuse: bool = True,
        use_kmm: bool = True,
        **kwargs: Any,
    ) -> T2TAttentionMetadata:
        return T2TAttentionMetadata(
            current_timestep=int(step_index),
            step_index=int(step_index),
            total_steps=int(total_steps),
            profile_name=str(profile_name),
            ns_base=None if ns_base is None else int(ns_base),
            dense_prefix_steps=(
                None if dense_prefix_steps is None else int(dense_prefix_steps)
            ),
            cq=None if cq is None else int(cq),
            ck=None if ck is None else int(ck),
            step_profile=(
                None if step_profile is None else tuple(map(float, step_profile))
            ),
            layer_profile=(
                None if layer_profile is None else tuple(map(float, layer_profile))
            ),
            cfg_mult=None if cfg_mult is None else tuple(map(float, cfg_mult)),
            ns_table=None if ns_table is None else str(ns_table),
            warm_iters=int(warm_iters),
            mask_refresh=int(mask_refresh),
            cfg_reuse=bool(cfg_reuse),
            use_kmm=bool(use_kmm),
            cache=cache,
        )


class T2TAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.T2T_ATTN

    @staticmethod
    def get_impl_cls() -> type["T2TAttentionImpl"]:
        return T2TAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[T2TAttentionMetadata]:
        return T2TAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[T2TAttentionMetadataBuilder]:
        return T2TAttentionMetadataBuilder


@dataclass
class _T2TRuntime:
    module: ModuleType
    deploy: Any
    key: tuple[Any, ...]
    profile: T2TProfile
    ns_map: dict[tuple[int, int, int], int]
    ns_tensors: dict[int, torch.Tensor]
    active_cache_id: int | None = None


def _profile_num_layers(profile: T2TProfile) -> int:
    if profile.num_heads == 12:
        return 30
    if profile.num_heads == 40:
        return 40
    if profile.num_heads == 56:
        return 50
    raise ValueError(
        f"No audited Token2Tile layer-profile geometry for H={profile.num_heads}"
    )


def _validate_profile(profile: T2TProfile) -> None:
    num_layers = _profile_num_layers(profile)
    if len(profile.step_profile) != 4 or len(profile.layer_profile) not in (
        6,
        num_layers,
    ):
        raise ValueError(f"Invalid Token2Tile profile shape: {profile.name}")
    if len(profile.cfg_mult) != 2:
        raise ValueError(f"Invalid Token2Tile CFG profile shape: {profile.name}")
    if not (0 < profile.ns_base <= profile.seq_len):
        raise ValueError(f"Invalid Token2Tile ns_base: {profile.name}")
    if profile.dense_prefix_steps < 0 or profile.cq <= 0 or profile.ck <= 0:
        raise ValueError(f"Invalid Token2Tile profile constants: {profile.name}")


def _resolve_profile(
    metadata: T2TAttentionMetadata,
    seq_len: int,
    num_heads: int,
) -> T2TProfile:
    manual_values = (
        metadata.ns_base,
        metadata.dense_prefix_steps,
        metadata.cq,
        metadata.ck,
        metadata.step_profile,
        metadata.layer_profile,
        metadata.cfg_mult,
    )
    if metadata.profile_name == "manual":
        if any(value is None for value in manual_values):
            raise ValueError(
                "t2t_profile='manual' requires ns_base, dense_prefix_steps, "
                "cq, ck, step_profile, layer_profile, and cfg_mult"
            )
        profile = T2TProfile(
            name="manual",
            num_heads=num_heads,
            seq_len=seq_len,
            total_steps=metadata.total_steps,
            ns_base=int(metadata.ns_base),
            dense_prefix_steps=int(metadata.dense_prefix_steps),
            cq=int(metadata.cq),
            ck=int(metadata.ck),
            step_profile=tuple(metadata.step_profile),  # type: ignore[arg-type]
            layer_profile=tuple(metadata.layer_profile),  # type: ignore[arg-type]
            cfg_mult=tuple(metadata.cfg_mult),  # type: ignore[arg-type]
        )
    else:
        if any(value is not None for value in manual_values):
            raise ValueError(
                "Token2Tile policy overrides require t2t_profile='manual'; audited "
                "profiles cannot be partially overridden"
            )
        if metadata.profile_name == "auto":
            matches = [
                candidate
                for candidate in _WAN_PROFILES.values()
                if candidate.num_heads == num_heads
                and candidate.seq_len == seq_len
                and candidate.total_steps == metadata.total_steps
            ]
            if len(matches) != 1:
                supported = ", ".join(sorted(_WAN_PROFILES))
                raise ValueError(
                    "No unique audited Token2Tile profile for "
                    f"H={num_heads}, N={seq_len}, steps={metadata.total_steps}. "
                    "Supported: "
                    f"{supported}"
                )
            profile = matches[0]
        else:
            try:
                profile = _WAN_PROFILES[metadata.profile_name]
            except KeyError as exc:
                supported = ", ".join(sorted(_WAN_PROFILES))
                raise ValueError(
                    f"Unknown Token2Tile profile {metadata.profile_name!r}. "
                    f"Supported: auto, manual, {supported}"
                ) from exc
            if (
                profile.num_heads != num_heads
                or profile.seq_len != seq_len
                or profile.total_steps != metadata.total_steps
            ):
                raise ValueError(
                    f"Token2Tile profile {profile.name} expects H={profile.num_heads}, "
                    f"N={profile.seq_len}, steps={profile.total_steps}; got "
                    f"H={num_heads}, N={seq_len}, steps={metadata.total_steps}"
                )
    _validate_profile(profile)
    return profile


def _step_window(step: int) -> int:
    # Thresholds for a 50-step schedule. The budget is read only on rebuild calls, so with a 12-step dense
    # prefix and R = 4 the windows in effect are steps 12-23, 24-31, 32-43, 44-end (0-based).
    if step < 21:
        return 0
    if step < 31:
        return 1
    if step < 41:
        return 2
    return 3


def _layer_bucket(layer_idx: int, profile: T2TProfile) -> int:
    """Match the archived per-model fractional six-bucket policy exactly."""
    # The audited Wan profiles cover the 30-layer 1.3B and 40-layer 14B
    # checkpoints.  Head count uniquely identifies those two architectures.
    num_layers = _profile_num_layers(profile)
    return min(layer_idx * 6 // num_layers, 5)


def _layer_factor(layer_idx: int, profile: T2TProfile) -> float:
    num_layers = _profile_num_layers(profile)
    if not 0 <= layer_idx < num_layers:
        raise ValueError(
            f"Token2Tile layer index {layer_idx} is outside [0, {num_layers})"
        )
    if len(profile.layer_profile) == num_layers:
        return profile.layer_profile[layer_idx]
    return profile.layer_profile[_layer_bucket(layer_idx, profile)]


def _load_ns_table(
    path: str,
    *,
    expected_layers: int,
    max_ns: int,
) -> tuple[tuple[tuple[int, ...], ...], str, dict[str, Any]]:
    """Load an exact per-(window, layer) ns table (schema t2t_ns_table_v1).

    Returns (rows, sha256, header). Head-count / sequence-length checks stay
    with the caller: the Wan path requires exact equality while the H3 path
    tolerates per-prompt used-token jitter.
    """
    table_path = Path(path).resolve()
    digest = _sha256(table_path)
    data = json.loads(table_path.read_text())
    if data.get("schema") != "t2t_ns_table_v1":
        raise ValueError(f"Unsupported Token2Tile ns-table schema in {table_path}")
    rows = tuple(tuple(int(v) for v in row) for row in data["ns"])
    if len(rows) != 4 or any(len(row) != expected_layers for row in rows):
        raise ValueError(f"Token2Tile ns-table {table_path} must be 4x{expected_layers}")
    if any(not 0 < v <= max_ns for row in rows for v in row):
        raise ValueError(
            f"Token2Tile ns-table {table_path} has ns outside (0, {max_ns}]"
        )
    return rows, digest, data


def _get_runtime(
    metadata: T2TAttentionMetadata,
    profile: T2TProfile,
    heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    total_heads: int | None = None,
) -> _T2TRuntime:
    # Under Ulysses each rank owns heads/ulysses_degree heads over the full
    # sequence, so the per-rank deploy is built with the local head count while
    # the audited ns-table geometry is checked against the model's total heads.
    total_heads = heads if total_heads is None else int(total_heads)
    ns_table = None
    if metadata.ns_table is not None:
        rows, digest, header = _load_ns_table(
            metadata.ns_table,
            expected_layers=_profile_num_layers(profile),
            max_ns=seq_len,
        )
        if int(header["heads"]) != total_heads or int(header["seq_len"]) != seq_len:
            raise ValueError(
                f"Token2Tile ns-table geometry mismatch: table H={header['heads']} "
                f"N={header['seq_len']} vs runtime H={total_heads} "
                f"(local {heads}) N={seq_len}"
            )
        ns_table = (rows, digest)
    key = (
        heads,
        total_heads,
        seq_len,
        head_dim,
        profile.name,
        profile.ns_base,
        profile.dense_prefix_steps,
        profile.cq,
        profile.ck,
        profile.step_profile,
        profile.layer_profile,
        profile.cfg_mult,
        metadata.mask_refresh,
        metadata.cfg_reuse,
        metadata.use_kmm,
        str(device),
        metadata.ns_table,
        None if ns_table is None else ns_table[1],
    )
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        module = _load_operator("wan")
        deploy = module.MaskedStepDeploy(
            heads,
            seq_len,
            profile.cq,
            C=profile.ck,
            D=head_dim,
            s_frac=profile.ns_base / seq_len,
            mask_refresh=metadata.mask_refresh,
            cfg_reuse=metadata.cfg_reuse,
        )
        backend_cfg = getattr(get_global_server_args(), "attention_backend_config", None) or {}
        if bool(backend_cfg.get("t2t_fast_init", False)):
            # t2t_fast_init also applies to the generic (Wan) path: same
            # init_state contract (Uq/Uk/ck/cq) as the H3 deploy.
            deploy.init_state = _make_fast_init_state(deploy, module)
        num_layers = _profile_num_layers(profile)
        ns_map = {
            (window, layer, cfg): min(
                seq_len,
                max(
                    1,
                    int(
                        round(
                            profile.ns_base
                            * profile.step_profile[window]
                            * _layer_factor(layer, profile)
                            * profile.cfg_mult[cfg]
                        )
                    ),
                ),
            )
            for window in range(4)
            for layer in range(num_layers)
            for cfg in range(2)
        }
        if ns_table is not None:
            # Exact table: replaces the separable ns_base x step x layer
            # product for both CFG parities (cfg_mult is intentionally not
            # applied on top of exact tables).
            ns_map = {
                (window, layer, cfg): ns_table[0][window][layer]
                for window in range(4)
                for layer in range(num_layers)
                for cfg in range(2)
            }
        ns_tensors = {
            value: torch.tensor(value, device=device, dtype=deploy.ns_t.dtype)
            for value in set(ns_map.values())
        }
        runtime = _T2TRuntime(
            module=module,
            deploy=deploy,
            key=key,
            profile=profile,
            ns_map=ns_map,
            ns_tensors=ns_tensors,
        )
        _RUNTIMES[key] = runtime
        logger.info(
            "Initialized Token2Tile runtime: profile=%s H=%d/%d N=%d D=%d "
            "CQ=%d CK=%d ns_base=%d prefix=%d R=%d cfg_reuse=%s kmm=%s "
            "ns_table=%s",
            profile.name,
            heads,
            total_heads,
            seq_len,
            head_dim,
            profile.cq,
            profile.ck,
            profile.ns_base,
            profile.dense_prefix_steps,
            metadata.mask_refresh,
            metadata.cfg_reuse,
            metadata.use_kmm,
            "-" if ns_table is None else ns_table[1][:12],
        )
        return runtime


def _sanitize_state(
    state: dict[str, torch.Tensor], previous: dict[str, torch.Tensor]
) -> None:
    for name in ("ck", "cq"):
        new = state[name]
        old = previous[name]
        bad = (~torch.isfinite(new.float()).all(-1)) | (
            new.float().abs().amax(-1) == 0
        )
        state[name] = torch.where(bad[..., None], old, new).contiguous()


def _h3_eigh_ridge(matrix: torch.Tensor):
    scale = float(matrix.diagonal().mean().clamp_min(1e-12))
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    for epsilon in (0.0, 1e-10, 1e-8, 1e-6):
        try:
            return torch.linalg.eigh(matrix + epsilon * scale * eye)
        except torch._C._LinAlgError:
            continue
    values, vectors = torch.linalg.eigh(matrix.cpu())
    return values.to(matrix.device), vectors.to(matrix.device)


def _h3_online_u(qf: torch.Tensor, kf: torch.Tensor, r: int = 32):
    uq, uk = [], []
    for head in range(qf.shape[0]):
        q, k = qf[head].double(), kf[head].double()
        gq = (q.T @ q) / q.shape[0]
        gk = (k.T @ k) / k.shape[0]
        eq, vq = _h3_eigh_ridge(gq)
        ek, vk = _h3_eigh_ridge(gk)
        eq, ek = eq.clamp_min(1e-8), ek.clamp_min(1e-8)
        pq = vq @ torch.diag(eq.sqrt()) @ vq.T
        pk = vk @ torch.diag(ek.sqrt()) @ vk.T
        pqi = vq @ torch.diag(eq.rsqrt()) @ vq.T
        pki = vk @ torch.diag(ek.rsqrt()) @ vk.T
        matrix = pq.T @ pk
        try:
            u, singular, vh = torch.linalg.svd(matrix)
        except torch._C._LinAlgError:
            u, singular, vh = torch.linalg.svd(matrix.cpu())
            u, singular, vh = u.to(matrix.device), singular.to(matrix.device), vh.to(matrix.device)
        root = singular[:r].sqrt()
        uq.append((pqi.T @ u[:, :r] * root).float())
        uk.append((pki @ vh[:r].T * root).float())
    return torch.stack(uq), torch.stack(uk)


@dataclass
class _H3Runtime:
    deploy: Any
    heads: int
    ns_map: dict[tuple[int, int], int]
    ns_tensors: dict[int, torch.Tensor]
    used_tokens: int
    video_tokens: int
    context_tokens: int
    mask_refresh: int
    cfg_reuse: bool
    dense_prefix: int


_H3_RUNTIMES: dict[tuple[Any, ...], _H3Runtime] = {}
_H3_RUNTIME_LOCK = threading.Lock()
_H3_SHARED_CACHE = T2TRequestCache()


def _h3_config() -> dict[str, Any]:
    cfg = getattr(get_global_server_args(), "attention_backend_config", None) or {}
    step_profile = tuple(
        float(v)
        for v in cfg.get(
            "t2t_step_profile", (1.574482, 0.985902, 0.555188, 0.884428)
        )
    )
    layer_profile = tuple(
        float(v)
        for v in cfg.get(
            "t2t_layer_profile",
            (1.189098, 1.123097, 1.278643, 0.915148, 0.956427, 0.537588),
        )
    )
    if len(step_profile) != 4 or len(layer_profile) != 6:
        raise ValueError("H3 Token2Tile needs 4 step and 6 layer multipliers")
    return {
        "video_tokens": int(cfg.get("t2t_h3_video_tokens", 0)),
        "s_frac": float(cfg.get("t2t_h3_s_frac", 0.05)),
        "dense_prefix": int(cfg.get("t2t_dense_prefix_steps", 12)),
        "mask_refresh": max(1, int(cfg.get("t2t_mask_refresh", 1))),
        # H3 has one coupled video/audio branch rather than Wan's two CFG
        # branches.  Keep CFG reuse independently configurable so R=1 really
        # means reuse-off; temporal R>1 remains controlled by mask_refresh.
        "cfg_reuse": bool(cfg.get("t2t_cfg_reuse", False)),
        "cq": cfg.get("t2t_cq", "auto"),
        "ck": int(cfg.get("t2t_ck", 2048)),
        "step_profile": step_profile,
        "layer_profile": layer_profile,
        "ns_table": cfg.get("t2t_ns_table"),
        "warm_iters": int(cfg.get("t2t_warm_iters", 4)),
        # t2t_persist_state: keep the whitening basis + k-means centroids of
        # the previous request (same geometry) as the warm start of the next
        # one instead of re-estimating them at the first sparse step.  Masks /
        # snapshots are still rebuilt per request (reset_cfg_reuse).  Serving
        # optimisation for the 8-way Ulysses regime where init_state is a
        # large share of a short request.
        "persist_state": bool(cfg.get("t2t_persist_state", False)),
        # fl2va keyframe rows: "pin" = exact context (default), "sparse" = join
        # the generated video in clustering/selection (text/audio stay pinned).
        "cond_mode": str(cfg.get("t2t_h3_cond_mode", "pin")),
        "fast_init": bool(cfg.get("t2t_fast_init", False)),
        # diagnostic: host-synchronise after every attention call of the first
        # sparse step (keeps Ulysses ranks in lockstep around the one-off init).
        "sync_first_sparse": bool(cfg.get("t2t_sync_first_sparse", False)),
        "sync_init": bool(cfg.get("t2t_sync_init", False)),
    }


def _h3_video_tokens_from_layout(used_tokens: int) -> int:
    """Video token count of the packed main-DiT document, derived from the
    modality layout the MiniMax-H3 DiT publishes on the forward context.
    Returns 0 when this call is not the main packed document (prompt refiner,
    direct callers) or the video segment is not the contiguous tail that the
    force-col path assumes."""
    try:
        layout = getattr(get_forward_context(), "minimax_h3_layout", None)
    except AssertionError:
        return 0
    if not layout or layout["video"] <= 0:
        return 0
    video = int(layout["video"])
    # fl2va: keyframe condition rows precede audio, so only the generated tail
    # is sparse; condition rows join text/audio as pinned exact context.
    tail = int(layout.get("video_tail", video))
    if layout["video_stop"] != used_tokens or tail <= 0:
        return 0
    if layout["video_start"] != used_tokens - video and tail == video:
        return 0
    return tail


def _h3_cond_sparse_rows(used_tokens: int, device: torch.device):
    """cond_mode='sparse': (video_tokens, (video_rows, context_rows)) where the
    video rows are [keyframe cond rows | generated tail]; None if no cond rows."""
    try:
        layout = getattr(get_forward_context(), "minimax_h3_layout", None)
    except AssertionError:
        return None
    if not layout or "video_tail" not in layout:
        return None
    video, tail = int(layout["video"]), int(layout["video_tail"])
    cond = video - tail
    if cond <= 0 or layout["video_stop"] != used_tokens:
        return None
    c0 = int(layout["video_start"])
    vid = torch.cat((torch.arange(c0, c0 + cond), torch.arange(used_tokens - tail, used_tokens)))
    ctx = torch.cat((torch.arange(0, c0), torch.arange(c0 + cond, used_tokens - tail)))
    return video, (vid.to(device), ctx.to(device))


def _get_h3_runtime(
    used_tokens: int,
    device: torch.device,
    heads: int = 28,
    video_tokens: int | None = None,
) -> _H3Runtime:
    """``heads`` = heads per kernel chunk on this rank (28 for the archived
    single-GPU 2x28 split; 56/ulysses_degree when Ulysses shards heads).
    ``video_tokens`` overrides the config constant (auto-derived layout)."""
    cfg = _h3_config()
    video_tokens = cfg["video_tokens"] if video_tokens is None else int(video_tokens)
    if not 0 < video_tokens < used_tokens:
        raise ValueError(
            f"t2t_h3_video_tokens must be in (0, {used_tokens}), got {video_tokens}"
        )
    context_tokens = used_tokens - video_tokens
    cq = (
        max(64, int(round(used_tokens / 128 / 4)) * 4)
        if cfg["cq"] in (None, "auto")
        else int(cfg["cq"])
    )
    ns_table = None
    if cfg["ns_table"] is not None:
        rows, digest, header = _load_ns_table(
            cfg["ns_table"], expected_layers=50, max_ns=used_tokens
        )
        # H3 used-token counts jitter by a few tokens per prompt; accept the
        # table within a small tolerance instead of exact sequence equality.
        # The archived H3 tables were sealed with the single-GPU 28-head chunk
        # geometry; the budgets are per (window, layer) and head-agnostic, so
        # any chunking of the same 56-head model is accepted.
        if int(header["heads"]) not in (28, 56) or abs(int(header["seq_len"]) - used_tokens) > 64:
            raise ValueError(
                f"Token2Tile H3 ns-table geometry mismatch: table H={header['heads']} "
                f"N={header['seq_len']} vs runtime H=56 (chunk {heads}) N={used_tokens}"
            )
        ns_table = (rows, digest)
    key = (
        heads,
        used_tokens,
        video_tokens,
        cfg["s_frac"],
        cfg["dense_prefix"],
        cfg["mask_refresh"],
        cfg["cfg_reuse"],
        cq,
        cfg["ck"],
        cfg["step_profile"],
        cfg["layer_profile"],
        str(device),
        cfg["ns_table"],
        None if ns_table is None else ns_table[1],
    )
    with _H3_RUNTIME_LOCK:
        # A runtime holds 1.7-3.4 GB per rank; keep only the current packed length.
        stale = [k for k in _H3_RUNTIMES if k[1] != used_tokens]
        for k in stale:
            _H3_RUNTIMES.pop(k)
        if stale:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        runtime = _H3_RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        module = _load_operator("h3")
        deploy = module.MaskedStepDeploy(
            heads,
            used_tokens,
            cq,
            C=cfg["ck"],
            D=128,
            r=32,
            s_frac=cfg["s_frac"],
            mask_refresh=cfg["mask_refresh"],
            cfg_reuse=cfg["cfg_reuse"],
            pin_tail=context_tokens,
        )
        if cfg["fast_init"]:
            deploy.init_state = _make_fast_init_state(deploy, module)
        ns_base = int(cfg["s_frac"] * used_tokens)
        ns_map = {
            (window, layer): min(
                used_tokens,
                max(
                    1,
                    int(
                        round(
                            ns_base
                            * cfg["step_profile"][window]
                            * cfg["layer_profile"][min(layer * 6 // 50, 5)]
                        )
                    ),
                ),
            )
            for window in range(4)
            for layer in range(50)
        }
        if ns_table is not None:
            ns_map = {
                (window, layer): min(used_tokens, ns_table[0][window][layer])
                for window in range(4)
                for layer in range(50)
            }
        ns_tensors = {
            value: torch.tensor(value, device=device, dtype=deploy.ns_t.dtype)
            for value in set(ns_map.values())
        }
        runtime = _H3Runtime(
            deploy=deploy,
            heads=heads,
            ns_map=ns_map,
            ns_tensors=ns_tensors,
            used_tokens=used_tokens,
            video_tokens=video_tokens,
            context_tokens=context_tokens,
            mask_refresh=cfg["mask_refresh"],
            cfg_reuse=cfg["cfg_reuse"],
            dense_prefix=cfg["dense_prefix"],
        )
        _H3_RUNTIMES[key] = runtime
        logger.info(
            "Initialized H3 Token2Tile force-col runtime: heads/chunk=%d used=%d "
            "video=%d ctx=%d CQ=%d s=%.4f prefix=%d R=%d cfg_reuse=%s",
            heads,
            used_tokens,
            video_tokens,
            context_tokens,
            cq,
            cfg["s_frac"],
            cfg["dense_prefix"],
            cfg["mask_refresh"],
            cfg["cfg_reuse"],
        )
        return runtime


class T2TAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args: Any,
    ) -> None:
        if causal:
            raise ValueError("Token2Tile does not support causal attention")
        if head_size != 128:
            raise ValueError("Token2Tile is source-locked to head dimension 128")
        if num_kv_heads not in (None, num_heads):
            raise ValueError("Token2Tile does not support grouped-query attention")
        if not math.isclose(softmax_scale, 1.0 / math.sqrt(128), rel_tol=1e-6):
            raise ValueError("Token2Tile has a fixed 1/sqrt(128) attention scale")
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.layer_idx = self._get_layer_idx(prefix)
        self._h3_states: dict[int, dict[str, torch.Tensor]] = {}
        self._dense = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    @staticmethod
    def _get_layer_idx(prefix: str) -> int:
        match = re.search(r"(?:blocks|transformer_blocks)\.(\d+)", prefix)
        if match is not None:
            return int(match.group(1))
        parts = prefix.split(".")
        if len(parts) < 3:
            raise ValueError(f"Invalid Token2Tile attention layer prefix: {prefix}")
        return int(parts[-3])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T2TAttentionMetadata,
    ) -> torch.Tensor:
        if query.ndim != 4:
            raise ValueError("Token2Tile expects rank-4 SGL attention tensors")
        profile = _resolve_profile(
            attn_metadata, int(query.shape[1]), self.num_heads
        )
        if attn_metadata.step_index < profile.dense_prefix_steps:
            return self._dense.forward(
                query, key, value, FlashAttentionMetadata()
            )
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Token2Tile requires equal self-attention Q/K/V shapes")
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError(
                "Token2Tile currently requires one request/CFG branch per call"
            )
        _, seq_len, heads, head_dim = query.shape
        ulysses = _ulysses_world_size()
        if (
            self.num_heads % ulysses
            or heads != self.num_heads // ulysses
            or head_dim != self.head_size
        ):
            raise ValueError(
                "Token2Tile Q/K/V geometry changed after initialization: got "
                f"H={heads} D={head_dim}, expected H={self.num_heads}/{ulysses} "
                f"D={self.head_size}"
            )
        if seq_len < max(profile.ck, profile.cq):
            raise ValueError("Token2Tile cluster count exceeds the token sequence length")

        runtime = _get_runtime(
            attn_metadata,
            profile,
            heads,
            seq_len,
            head_dim,
            query.device,
            total_heads=self.num_heads,
        )
        cache = attn_metadata.cache
        if cache.runtime_key is None:
            cache.runtime_key = runtime.key
        elif cache.runtime_key != runtime.key:
            raise RuntimeError("Token2Tile request geometry changed during denoising")
        cache_id = id(cache)
        if runtime.active_cache_id != cache_id:
            runtime.deploy.reset_cfg_reuse()
            runtime.active_cache_id = cache_id
            cache.initialized_runtime = True

        call_key = (attn_metadata.step_index, self.layer_idx)
        parity = cache.call_counts.get(call_key, 0)
        if parity > 1:
            raise RuntimeError(
                "Token2Tile observed more than two CFG branches for one layer/step"
            )
        cache.call_counts[call_key] = parity + 1

        # SGL's model forward runs under BF16 autocast.  Token2Tile's archived
        # Triton/CUDA ABI is strictly FP16, including intermediate bmm output,
        # so fence the complete source-locked region from outer autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            q = query[0].transpose(0, 1).to(torch.float16).contiguous()
            k = key[0].transpose(0, 1).to(torch.float16).contiguous()
            v = value[0].transpose(0, 1).to(torch.float16).contiguous()
            deploy = runtime.deploy
            current = torch.cuda.current_stream(query.device)
            deploy.s_main.wait_stream(current)
            deploy.s_side.wait_stream(current)
            deploy.set_cfg_call(parity, self.layer_idx, attn_metadata.step_index)
            ns = runtime.ns_map[
                (
                    _step_window(attn_metadata.step_index),
                    self.layer_idx,
                    parity,
                )
            ]
            with torch.cuda.stream(deploy.s_main):
                deploy.ns_t.copy_(runtime.ns_tensors[ns])

            state_key = (self.layer_idx, parity)
            state = cache.states.get(state_key)
            if state is None:
                state = deploy.init_state(
                    q,
                    k,
                    warm_iters=attn_metadata.warm_iters,
                    seed=self.layer_idx * 2 + parity,
                )
                cache.states[state_key] = state
            previous = {name: state[name].clone() for name in ("ck", "cq")}
            output = deploy.step(q, k, v, state)
            _sanitize_state(state, previous)
        sparse_output = (
            output.transpose(0, 1).unsqueeze(0).to(query.dtype).contiguous()
        )
        return sparse_output

    @staticmethod
    def _current_step() -> int:
        try:
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )

            return int(get_forward_context().current_timestep)
        except AssertionError:
            return 0

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """MiniMax-H3 packed [text|audio|video|pad] force-col path."""
        if self.num_heads != 56:
            return self._dense.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        step = self._current_step()
        cfg = _h3_config()
        if step < cfg["dense_prefix"]:
            return self._dense.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        used = int(max_seqlen)
        if used <= 0 or used > query.shape[0]:
            raise ValueError(f"Invalid H3 packed used length {used}/{query.shape[0]}")
        # MiniMax-H3 resolves one transformer backend for both the main DiT
        # blocks and its short prompt-token refiner.  Token2Tile is defined only
        # for the packed main-DiT document, whose configured video segment is
        # strictly shorter than the full used sequence.  Keep auxiliary
        # short-sequence attention on the same dense FA path as the baseline.
        # In particular, the prompt refiner has only a handful of tokens and
        # must never be interpreted as an H3 video document.
        video_tokens = cfg["video_tokens"]
        if video_tokens <= 0:
            # t2t_h3_video_tokens unset/0 -> derive from the DiT's published
            # packed layout (per-duration geometry without config constants).
            video_tokens = _h3_video_tokens_from_layout(used)
        rows = None
        if cfg["cond_mode"] == "sparse":
            sparse = _h3_cond_sparse_rows(used, query.device)
            if sparse is not None:
                video_tokens, rows = sparse
        if video_tokens <= 0 or video_tokens >= used:
            return self._dense.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        # Ulysses hands each rank 56/ulysses_degree heads over the full packed
        # sequence.  Keep the archived 28-head kernel chunk when it divides the
        # local head count; otherwise one chunk of all local heads.
        local_heads = int(query.shape[1])
        ulysses = _ulysses_world_size()
        if local_heads * ulysses != self.num_heads:
            raise ValueError(
                f"H3 Token2Tile head sharding mismatch: local H={local_heads} x "
                f"ulysses {ulysses} != {self.num_heads}"
            )
        heads_per_chunk = 28 if local_heads % 28 == 0 else local_heads
        num_chunks = local_heads // heads_per_chunk
        runtime = _get_h3_runtime(
            used, query.device, heads=heads_per_chunk, video_tokens=video_tokens
        )
        context = runtime.context_tokens
        video = runtime.video_tokens
        if context > video:
            raise ValueError("H3 Token2Tile dummy query padding requires video >= context")
        if step == runtime.dense_prefix:
            if cfg["persist_state"]:
                # keep only states matching this request's chunk geometry
                keep = {
                    chunk: st
                    for chunk, st in self._h3_states.items()
                    if st["Uq"].shape[0] == heads_per_chunk
                    and st["cq"].shape[1] == runtime.deploy.CQ
                    and st["ck"].shape[1] == runtime.deploy.C
                }
                if self.layer_idx == 0:
                    logger.info(
                        "Token2Tile H3 persist: layer0 had %d state(s), keeping %d "
                        "(heads/chunk=%d CQ=%d C=%d, impl id=%d)",
                        len(self._h3_states),
                        len(keep),
                        heads_per_chunk,
                        runtime.deploy.CQ,
                        runtime.deploy.C,
                        id(self),
                    )
                self._h3_states = keep
            else:
                self._h3_states.clear()
            if self.layer_idx == 0:
                runtime.deploy.reset_cfg_reuse()

        q_used = query[:used]
        k_used = key[:used]
        v_used = value[:used]
        qh = q_used.transpose(0, 1)
        kh = k_used.transpose(0, 1)
        vh = v_used.transpose(0, 1)
        deploy = runtime.deploy
        outputs = []
        with torch.autocast(device_type="cuda", enabled=False):
            for chunk in range(num_chunks):
                h0, h1 = chunk * heads_per_chunk, (chunk + 1) * heads_per_chunk
                if rows is None:
                    q_video = qh[h0:h1, context:]
                    k_reordered = torch.cat(
                        (kh[h0:h1, context:], kh[h0:h1, :context]), dim=1
                    )
                    v_reordered = torch.cat(
                        (vh[h0:h1, context:], vh[h0:h1, :context]), dim=1
                    )
                else:
                    vid_rows, ctx_rows = rows
                    q_video = qh[h0:h1].index_select(1, vid_rows)
                    k_reordered = torch.cat(
                        (kh[h0:h1].index_select(1, vid_rows), kh[h0:h1].index_select(1, ctx_rows)), dim=1
                    )
                    v_reordered = torch.cat(
                        (vh[h0:h1].index_select(1, vid_rows), vh[h0:h1].index_select(1, ctx_rows)), dim=1
                    )
                if (
                    float(q_video.float().abs().amax()) > 8192.0
                    or float(k_reordered.float().abs().amax()) > 8192.0
                    or float(v_reordered.float().abs().amax()) > 8192.0
                ):
                    raise ValueError("H3 Token2Tile fp16-range guard triggered")
                q_kernel = torch.cat(
                    (q_video, q_video[:, :context]), dim=1
                ).to(torch.float16).contiguous()
                k_kernel = k_reordered.to(torch.float16).contiguous()
                v_kernel = v_reordered.to(torch.float16).contiguous()
                current = torch.cuda.current_stream(query.device)
                deploy.s_main.wait_stream(current)
                deploy.s_side.wait_stream(current)
                deploy.set_cfg_call(
                    0, key=(self.layer_idx, chunk), step=step
                )
                ns = runtime.ns_map[(_step_window(step), self.layer_idx)]
                with torch.cuda.stream(deploy.s_main):
                    deploy.ns_t.copy_(runtime.ns_tensors[ns])
                state = self._h3_states.get(chunk)
                sync_init = cfg["sync_init"] and state is None
                if sync_init:
                    # first sparse call: host syncs around init/step (as the
                    # diagnostic profiler does) -- see OPTIMIZATION_RESULTS.md
                    torch.cuda.synchronize(query.device)
                if state is None:
                    state = deploy.init_state(
                        q_kernel,
                        k_kernel,
                        warm_iters=cfg["warm_iters"],
                        seed=self.layer_idx * 8 + chunk,
                    )
                    self._h3_states[chunk] = state
                    if sync_init:
                        torch.cuda.synchronize(query.device)
                previous = {
                    name: state[name].clone() for name in ("ck", "cq")
                }
                out = deploy.step(q_kernel, k_kernel, v_kernel, state)
                if sync_init:
                    torch.cuda.synchronize(query.device)
                _sanitize_state(state, previous)
                outputs.append(out[:, :video].float())

        video_output = torch.cat(outputs, dim=0).transpose(0, 1).to(query.dtype)
        q_cu = torch.tensor([0, context], dtype=torch.int32, device=query.device)
        kv_cu = torch.tensor([0, used], dtype=torch.int32, device=query.device)
        context_output = flash_attn_varlen_func(
            q_used[:context] if rows is None else q_used.index_select(0, rows[1]),
            k_used,
            v_used,
            cu_seqlens_q=q_cu,
            cu_seqlens_k=kv_cu,
            max_seqlen_q=context,
            max_seqlen_k=used,
            softmax_scale=self.softmax_scale,
            causal=False,
        )
        if isinstance(context_output, tuple):
            context_output = context_output[0]
        result = torch.zeros_like(query)
        if rows is None:
            result[:context] = context_output
            result[context:used] = video_output
        else:
            result[rows[1]] = context_output
            result[rows[0]] = video_output
        if cfg["sync_first_sparse"] and step == runtime.dense_prefix:
            torch.cuda.synchronize(query.device)
        return result