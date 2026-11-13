"""Pairwise video fidelity against the dense video: PSNR and SSIM (imageio-ffmpeg rgb24 frames,
PSNR from the mean per-frame MSE, skimage SSIM with data_range=255) and LPIPS (alex, frame mean,
[-1,1] input).

usage: metrics.py watch ROOT   -- loop until every run cell with a dense partner is scored
       metrics.py pair REF CAND OUT
"""
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity

_LP = None


def frames(path):
    stream = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", input_params=["-threads", "4"])
    meta = next(stream)
    width, height = meta["size"]
    try:
        for raw in stream:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        stream.close()


def pair(ref, cand):
    global _LP
    if _LP is None:
        _LP = lpips.LPIPS(net="alex", spatial=False, verbose=False).cuda().eval()
    mse, ssim, lp = [], [], []
    identical = True
    a, b = frames(ref), frames(cand)
    try:
        for x, y in itertools.zip_longest(a, b):
            assert x is not None and y is not None, "Different frame counts"
            assert x.shape == y.shape
            if np.array_equal(x, y):
                mse.append(0.0); ssim.append(1.0); lp.append(0.0); continue
            identical = False
            d = x.astype(np.float32) - y.astype(np.float32)
            mse.append(float(np.mean(d * d, dtype=np.float64)))
            ssim.append(float(structural_similarity(x, y, data_range=255, channel_axis=2)))
            with torch.inference_mode():
                tx = torch.from_numpy(np.ascontiguousarray(x)).cuda().permute(2, 0, 1)[None].float().div(127.5).sub(1)
                ty = torch.from_numpy(np.ascontiguousarray(y)).cuda().permute(2, 0, 1)[None].float().div(127.5).sub(1)
                lp.append(float(_LP(tx, ty).item()))
    finally:
        a.close(); b.close()
    m = statistics.mean(mse)
    return dict(frames=len(mse), pixels_identical=identical, decoder=imageio_ffmpeg.get_ffmpeg_version(),
                psnr_video=None if m == 0 else 10 * math.log10(255 ** 2 / m),
                psnr_frame_min=None if max(mse) == 0 else 10 * math.log10(255 ** 2 / max(mse)),
                ssim_mean=statistics.mean(ssim), lpips_mean=statistics.mean(lp))


def score(ref, cand, out):
    res = dict(reference=str(ref), candidate=str(cand),
               reference_sha256=hashlib.sha256(Path(ref).read_bytes()).hexdigest(),
               candidate_sha256=hashlib.sha256(Path(cand).read_bytes()).hexdigest(),
               quality=pair(ref, cand))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1) + "\n"); tmp.replace(out)


def watch(root, shard=0, nshard=1):
    root = Path(root)
    while True:
        todo = []
        for done in sorted(root.glob("runs/*/*/*/DONE")):
            cell = done.parent
            arm, node = cell.parent.name, cell.parent.parent.name
            if arm == "dense" or cell.name.startswith("_"):
                continue
            ref = root / "runs" / node / "dense" / cell.name / "video.mp4"
            if not (ref.parent / "DONE").exists():
                continue
            out = root / "metrics" / node / arm / f"{cell.name}.json"
            if not out.exists():
                todo.append((ref, cell / "video.mp4", out))
        # extra pairs: repeats listed in metrics/extra_pairs.json [[ref,cand,out],...]
        extra = root / "metrics" / "extra_pairs.json"
        if extra.exists():
            for r, c, o in json.loads(extra.read_text()):
                if Path(r).exists() and Path(c).exists() and not Path(o).exists():
                    todo.append((Path(r), Path(c), Path(o)))
        todo = [t for t in todo if int(hashlib.md5(str(t[2]).encode()).hexdigest(), 16) % nshard == shard]
        for ref, cand, out in todo:
            try:
                score(ref, cand, out); print("SCORED", out, flush=True)
            except Exception as e:  # keep the watcher alive
                print("FAILED", out, repr(e), flush=True)
        if (root / "metrics" / "STOP").exists():
            break
        if not todo:
            time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1] == "watch":
        watch(sys.argv[2], *(int(x) for x in sys.argv[3:5])) if len(sys.argv) > 4 else watch(sys.argv[2])
    else:
        score(*sys.argv[2:5])
