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

## Where the shipped settings come from

**Fixed choices.** The kernels fix the head dimension (128) and the number of key clusters (2048); the low-rank
space has rank 32. Query clusters hold about 128 queries each: `"auto"` gives roughly N/128 clusters on H3, and Wan
at 720p uses 512. The first 12 denoising steps stay dense and the structure is rebuilt every 4 steps
(`t2t_dense_prefix_steps`, `t2t_mask_refresh`); both were chosen in the H3 ablations of the paper and are used
unchanged on Wan.

**Step profile.** The four multipliers in `t2t_step_profile` come from an offline calibration that is not part of
this repository:

1. One calibration request per model and task is denoised with dense attention. At one probe step in each window
   (steps 17, 25, 37 and 45), the denoiser is replayed with sparse attention at eleven budgets from 0.5% to 12.5%,
   and the squared error of its output against the dense output is recorded. This gives an error curve per window.
2. For a target budget, dynamic programming picks one grid budget per window that minimises the summed error,
   weighting the windows by their lengths (12, 8, 12 and the remaining steps), such that the length-weighted mean
   does not exceed the target; the result is then scaled so that the mean equals the target.
3. The multipliers are the window budgets divided by the target budget.

| Config | Window budgets (%) | Source |
|---|---|---|
| `h3/t2t_i2v_1p25.json` | 2.0 / 1.0 / 0.75 / 1.0 | Calibration on H3 image-to-video, target 1.25% |
| `wan/t2v_t2t_2p5.json` | 5.0 / 2.0 / 1.0 / 1.0 | Calibration on Wan text-to-video, target 2.5% |
| `wan/i2v_t2t_2p5.json` | 5.0 / 2.5 / 0.75 / 1.0 | Calibration on Wan image-to-video, target 2.5% |
| `wan/{t2v,i2v}_t2t_5.json` | 7.5 / 5.0 / 2.5 / 5.0 | Calibration on Wan, target 5% (the same result for both tasks) |
| `h3/t2t_t2v_1p5.json`, `h3/t2t_5.json` | ratio 8 : 4 : 1 : 1 | A fixed schedule from earlier H3 experiments, scaled to the target budget; it is not the calibration result for these budgets |

**Window boundaries.** The backend applies the four multipliers to denoising steps 13-21, 22-31, 32-41 and 42 to
the end (1-based, after the 12 dense steps). The calibration weighted the windows as 12, 8, 12 and 5-6 steps, so
the budget averaged over the sparse steps is slightly below the nominal one: 1.34% for the 1.5% H3 config, 1.18%
for 1.25%, 2.23-2.28% for the 2.5% Wan configs, 4.93% for the 5% Wan configs and 4.48% for the 5% H3 config. The names of the configs and the
budgets quoted with the results are the nominal values.

**H3 at 5%.** `h3/t2t_5.json` serves both text-to-video and image-to-video.

## Request settings

Model, resolution, frame count, steps, guidance and parallelism (`num_gpus`, `ulysses_degree`) are standard
SGLang-Diffusion fields. The shipped configs use 8 GPUs with Ulysses sequence parallelism; set `num_gpus` and
`ulysses_degree` together for a different GPU count.
