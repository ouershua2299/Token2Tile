# Configuration

A config in `configs/` is an SGLang-Diffusion generation config. Token2Tile is selected with
`"attention_backend": "t2t_attn"` and tuned through `attention_backend_config`. The dense configs use
`"attention_backend": "fa"` (FlashAttention-3) and are otherwise identical apart from the warm-up length.

## Budget

The token budget of a call is

    budget = base x step_profile[window] x layer_profile[group]

rounded to whole tokens. `window` is one of four windows of the sparse denoising steps and `group` is one of six
equal groups of DiT layers. Each query keeps the shortest prefix of whole key clusters that reaches this budget.

| Key | Model | Meaning |
|---|---|---|
| `t2t_h3_s_frac` | H3 | Base budget as a fraction of the sequence (0.015 = 1.5%). |
| `t2t_ns_base` | Wan | Base budget in tokens (1890 = 2.5% of 75,600). The value is tied to the sequence length; recompute it for another resolution or frame count. |
| `t2t_step_profile` | both | Four multipliers, one per denoising window. The shipped values are the calibrated schedules of the paper; `[1, 1, 1, 1]` is a uniform schedule. |
| `t2t_layer_profile` | both | Multipliers per layer group: six values (H3 requires six; Wan also accepts one value per layer). All ones in the shipped configs; the paper allocates the budget over denoising windows only. |
| `t2t_cfg_mult` | Wan | Two multipliers for the conditional and unconditional branch. With R > 1 the unconditional branch reuses the selection of the conditional one, so only the first has an effect. |
| `t2t_profile` | Wan | `"manual"` (all shipped configs) takes the budget from the keys above, which are then all required. The default `"auto"` selects a built-in profile by head count, sequence length and step count and rejects those keys; the built-in profiles are not the settings of the paper. |
| `t2t_ns_table` | both | Optional JSON table of per-(window, layer) token budgets that replaces the product above; not used by the shipped configs. |

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
| `t2t_cq` | Number of query clusters. H3: `"auto"` gives N/128 rounded to a multiple of 4, with N the packed sequence length. Wan: an integer is required; the shipped 720p configs use 512. |
| `t2t_warm_iters` | k-means iterations when a request initialises its centroids at its first sparse step (4). Every rebuild then takes one warm-started update of the key clusters and two of the query clusters. |
| `t2t_fast_init` | Computes the projections and initial centroids of a request on the GPU without host synchronisation. |
| `t2t_batched_whitening` | Without `t2t_fast_init`: computes the projections P_q, P_k of all heads in one batched decomposition instead of a per-head loop. |
| `t2t_persist_state` | H3: keep projections and centroids across requests of the same shape (off). |
| `t2t_h3_video_tokens` | H3: number of video tokens in the packed sequence; 0 (shipped) derives it from the layout published by the DiT. |
| `t2t_sync_init`, `t2t_sync_first_sparse` | H3 diagnostics: host synchronisation around the first sparse step. |

## MiniMax-H3 context tokens

Text and audio tokens are packed with the video tokens. Their query rows use dense attention; their keys form one
reserved key cluster that every video query attends to exactly. For image-to-video, `t2t_h3_cond_mode: "sparse"`
lets the keyframe rows go through the sparse operator.

## Where the shipped settings come from

**Fixed choices.** The kernels fix the head dimension (128) and the number of key clusters (2048); the low-rank
space has rank 32. Query clusters hold about 128 queries each: `"auto"` gives roughly N/128 clusters on H3, and Wan
at 720p uses 512. The structure is rebuilt every 4 steps (`t2t_mask_refresh`), the default of the reuse
ablation in the paper, and the first 12 denoising steps stay dense (`t2t_dense_prefix_steps`). Both are used unchanged
on Wan. The H3 path has its own defaults for omitted keys; set every key explicitly, as the shipped configs do.

**Step profile.** The four multipliers in `t2t_step_profile` were chosen with an offline calibration that is not
part of this repository:

1. One calibration request per model and task is denoised with dense attention. At one probe step in each window
   (steps 17, 25, 37 and 45), the denoiser is replayed with sparse attention at eleven budgets from 0.5% to 12.5%,
   and the squared error of its output against the dense output is recorded. This gives an error curve per window.
2. For a target budget, dynamic programming picks one grid budget per window that minimises the summed error,
   weighting the windows by their lengths (12, 8, 12 and the remaining steps), such that the length-weighted mean
   does not exceed the target; the result is then scaled so that the mean equals the target.
3. The multipliers are the window budgets divided by the target budget.

| Config | Window budgets (%) |
|---|---|
| `h3/t2t_t2v_1p5.json` | 3.06 / 1.53 / 0.38 / 0.38 |
| `h3/t2t_i2v_1p25.json` | 2.0 / 1.0 / 0.75 / 1.0 |
| `h3/t2t_5.json` (both tasks) | 10.21 / 5.10 / 1.28 / 1.28 |
| `wan/t2v_t2t_2p5.json` | 5.05 / 2.02 / 1.01 / 1.01 |
| `wan/i2v_t2t_2p5.json` | 5.0 / 2.5 / 0.75 / 1.0 |
| `wan/{t2v,i2v}_t2t_5.json` | 7.5 / 5.0 / 2.5 / 5.0 |

**Window boundaries.** The backend switches multipliers at denoising steps 22, 32 and 42 (1-based), and a budget
takes effect when the structure is rebuilt, because reuse calls keep the selection of the last rebuild. With the
shipped settings (12 dense steps, R = 4) rebuilds fall on steps 13, 17, 21, 25, ..., so the four multipliers are in
effect for steps 13-24, 25-32, 33-44 and 45 to the end, the windows of the paper (12, 8, 12 and 5-6 steps), and the
budget averaged over the sparse steps equals the nominal one. With a different R or dense prefix the effective
windows shift accordingly.

## Request settings

Model, resolution, frame count, steps, guidance and parallelism (`num_gpus`, `ulysses_degree`) are standard
SGLang-Diffusion fields. The shipped configs use 8 GPUs with Ulysses sequence parallelism; set `num_gpus` and
`ulysses_degree` together for a different GPU count.
