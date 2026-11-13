# Configuration

A config in `configs/` is an SGLang-Diffusion generation config. Token2Tile is selected with
`"attention_backend": "t2t_attn"` and tuned through `attention_backend_config`. The dense configs use
`"attention_backend": "fa"` (FlashAttention-3) and are otherwise identical.

## Budget

The token budget of a call is

    budget = base x step_profile[window] x layer_profile[group]

rounded to whole tokens. `window` is one of four windows of the sparse denoising steps and `group` is one of six
equal groups of DiT layers. Each query keeps the shortest prefix of whole key clusters that reaches this budget.

| Key | Model | Meaning |
|---|---|---|
| `t2t_h3_s_frac` | H3 | Base budget as a fraction of the sequence (0.015 = 1.5%). |
| `t2t_ns_base` | Wan | Base budget in tokens (1890 = 2.5% of 75,600). |
| `t2t_step_profile` | both | Four multipliers, one per denoising window. The shipped values are the calibrated schedules of the paper; `[1, 1, 1, 1]` is a uniform schedule. |
| `t2t_layer_profile` | both | Six multipliers, one per layer group (or one per layer). |
| `t2t_cfg_mult` | Wan | Two multipliers for the conditional and unconditional branch. |
| `t2t_profile` | Wan | `"manual"` uses the keys above. |

To change the speed-quality trade-off, scale `t2t_h3_s_frac` or `t2t_ns_base`. A larger budget is slower and closer
to dense attention.

## Schedule

| Key | Meaning |
|---|---|
| `t2t_dense_prefix_steps` | Number of initial denoising steps that run dense FlashAttention (12). |
| `t2t_mask_refresh` | Reuse interval R: clusters, selections and unions are rebuilt every R denoising steps and reused in between (4). Outputs are never cached. |
| `t2t_cfg_reuse` | Share the structure between CFG branches when `t2t_mask_refresh` is 1. With R > 1 the unconditional branch always reuses the structure of the conditional one. |

## Clustering

| Key | Meaning |
|---|---|
| `t2t_ck` | Number of key clusters. The kernels require 2048. |
| `t2t_cq` | Number of query clusters; `"auto"` uses about N/128 (H3), 512 for Wan at 720p. |
| `t2t_warm_iters` | k-means iterations when a request initialises its centroids (4). Later calls take one warm-started update. |
| `t2t_fast_init`, `t2t_batched_whitening` | GPU-batched computation of the per-request projections and initial centroids. |
| `t2t_persist_state` | H3: keep projections and centroids across requests of the same shape (off). |

## MiniMax-H3 context tokens

Text and audio tokens are packed with the video tokens. Their query rows use dense attention; their keys form one
reserved key cluster that every video query attends to exactly. For image-to-video, `t2t_h3_cond_mode: "sparse"`
lets the keyframe rows go through the sparse operator.

## Request settings

Model, resolution, frame count, steps, guidance and parallelism (`num_gpus`, `ulysses_degree`) are standard
SGLang-Diffusion fields. The shipped configs use 8 GPUs with Ulysses sequence parallelism; set `num_gpus` and
`ulysses_degree` together for a different GPU count.
