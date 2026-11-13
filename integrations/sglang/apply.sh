#!/usr/bin/env bash
# Add the Token2Tile attention backend to an SGLang checkout at SGLANG_BASE_COMMIT.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); sgl=${1:?usage: apply.sh <sglang checkout>}
base=$(cat "$here/SGLANG_BASE_COMMIT")
[ "$(git -C "$sgl" rev-parse HEAD)" = "$base" ] || { echo "expected SGLang at $base" >&2; exit 1; }
git -C "$sgl" apply --check "$here/token2tile_sglang.patch"
git -C "$sgl" apply "$here/token2tile_sglang.patch"
cp "$here/backend/t2t_attn.py" "$sgl/python/sglang/multimodal_gen/runtime/layers/attention/backends/"
echo "Token2Tile backend installed. Select it with attention_backend=t2t_attn (pip install the token2tile package, or set T2T_OPERATOR_DIR)."
