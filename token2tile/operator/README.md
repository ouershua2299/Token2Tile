# Operator

`deploy_step.py` exposes `MaskedStepDeploy`. One `step(q, k, v, state)` call is one attention layer call on
`[H, N, 128]` FP16 tensors (H heads of one rank, N tokens). `init_state(q, k, ...)` builds the per-request low-rank
projections and the initial centroids.

## Stages of a call

| # | Stage | Where | Refresh call | Reuse call |
|---|---|---|---|---|
| 1 | Project Q, K to 32-D | cuBLAS `bmm` in `deploy_step.py` | Q and K | Q only |
| 2 | Query and key clustering, value sums, cluster-ordered V | `kernels/kmeans_maintain` | yes | value sums and V only |
| 3 | Cluster scores `QU @ centroids` | cuBLAS `bmm` | yes | reused selection |
| 4 | Per-query budget selection and per-cluster union | `kernels/select_union` | yes | reused |
| 5 | Gather K into cluster order | `fast_permute.py` | yes | yes |
| 6 | Task plan, exact attention over the union in tiles, per-query residual over the other clusters, merge | `kernels/tile_attn` | yes | yes (plan reused) |

Structure (labels, permutations, centroids, unions, task plan) is rebuilt every `mask_refresh` denoising steps (4 in
the paper) and reused in between; outputs are never cached. `fast_assign.py` and `fused_update_vstats.py` are used by
`init_state` for the initial centroids.

## CUDA kernels (`kernels/<name>/src`)

| Kernel | Entry points used | Inputs -> outputs |
|---|---|---|
| `tile_attn` | `run`, `run_planned` | original-order `q`, cluster-ordered `kp`, `vp`, cluster sizes, packed union bitmap, projected queries `QU`, scaled centroids, value sums, query labels/permutation -> attention output `[H, N, 128]`. `run_planned` keeps the task plan across calls (`rebuild_plan=False` on reuse). |
| `select_union` | `run_packed`, `run_split_packed` | cluster scores `gl [H, N, C_K]`, cluster sizes, per-query token budget `ns` -> packed union bitmap `[H, C_Q, C_K/8]` and per-row selected counts. The `split` variant caps rows per CUDA block for imbalanced query clusters. |
| `kmeans_maintain` | `run`, `run_vstats_gather` | projected `QU`, `KU`, values, previous centroids -> labels, updated centroids, counts, permutations, cluster-ordered values and value sums. `pin_tail` reserves the last key cluster for pinned context tokens. |

Each kernel is compiled on first use with `torch.utils.cpp_extension.load` for `sm_90a` (flags in `deploy_step.py`).

## Fixed limits

Head dimension 128; 2048 key clusters; Hopper (SM90) only; batch 1; non-causal attention; selector packing assumes
fewer than 524,288 tokens per rank.
