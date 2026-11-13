# SGLang-Diffusion integration

Base: the SGLang commit in `SGLANG_BASE_COMMIT`. `./apply.sh <sglang checkout>` applies the patch and copies the backend.

| File | Purpose |
|---|---|
| `backend/t2t_attn.py` | The `t2t_attn` attention backend. Wan2.1 (`forward`): built-in or manual step/layer budget profiles, per-request projections and cluster state, one call per CFG branch. MiniMax-H3 (`forward_varlen`): text/audio rows computed densely, their keys kept exact, video rows through the operator. Dense prefix steps call the stock FlashAttention backend. |
| `token2tile_sglang.patch` | Registers the backend: enum and resolver (`platforms/`), supported-backend list (`dits/base.py`), per-step metadata in the denoising stage, the packed modality layout published by the H3 DiT, and config normalisation in `server_args.py`. |
