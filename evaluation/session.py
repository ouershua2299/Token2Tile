"""Resident generation session: one backend configuration, many independent cells.

usage: python evaluation/session.py PLAN.json

PLAN.json: {"arm": <label>, "config": <configs/...json>, "cells": [{"name", "prompt", "seed", "output",
"request_overrides"?, "burnin"?}], "warmup"?: "none"}.  Per cell, an untimed warm-up copy of the identical request
(same prompt, seed and geometry, ``warmup_steps`` steps) runs first so that shape-specific JIT and runtime construction
land outside the timed request; the formal request is then timed by the framework (``perf.json`` ->
``total_duration_ms``).  The same protocol applies to every backend.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


class _SkipWarm(Exception):
    pass


def main():
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        SamplingParams,
        generate_request_id,
    )
    from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
        _resolve_cli_sampling_params_cls,
        add_multimodal_gen_generate_args,
        maybe_dump_performance,
    )
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import prepare_request
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.utils import FlexibleArgumentParser

    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    opts = ap.parse_args()
    plan = json.loads(Path(opts.plan).read_text())
    if plan.get("skip"):  # a plan marked "skip": true is ignored
        print("SESSION_SKIPPED", opts.plan, flush=True)
        return
    config = Path(plan["config"])
    data = json.loads(config.read_text())
    first = plan["cells"][0]
    parser = add_multimodal_gen_generate_args(FlexibleArgumentParser())
    ports = plan.get("ports", [39863, 19863, 49863])
    cli = [
        "--model-path", data["model_path"], "--config", str(config),
        "--prompt", first["prompt"], "--seed", str(first["seed"]), "--save-output",
        "--master-port", str(ports[0]), "--scheduler-port", str(ports[1]), "--port", str(ports[2]),
    ]
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], *cli]
    args, unknown = parser.parse_known_args(cli)
    args.request_id = "mocked_fake_id_for_offline_generate"
    server = ServerArgs.from_cli_args(args, unknown)
    cls = _resolve_cli_sampling_params_cls(server)
    fields = {f.name for f in dataclasses.fields(cls)}
    sampling = {
        k: v for k, v in ServerArgs.load_config_file(str(config)).items()
        if k in fields and v is not None
    }
    sampling.update(cls.get_cli_args(args))
    sys.argv = saved_argv
    warmup_steps = int(data.get("warmup_steps", 14))

    session_start = time.perf_counter()
    generator = DiffGenerator.from_pretrained(
        model_path=server.model_path, server_args=server, local_mode=True
    )
    startup_s = time.perf_counter() - session_start
    print("SESSION_LOADED", round(startup_s, 1), flush=True)
    prev_params = None
    try:
        for index, cell in enumerate(plan["cells"]):
            dest = Path(cell["output"])
            if (dest / "DONE").exists() and not cell.get("burnin"):
                print("CELL_SKIP_DONE", cell["name"], flush=True)
                continue
            if dest.exists():  # partial output from an interrupted session: keep it aside
                dest.rename(dest.with_name(dest.name + f".partial_{int(time.time())}"))
            dest.mkdir(parents=True, exist_ok=False)
            params = dict(
                sampling, prompt=cell["prompt"], seed=int(cell["seed"]),
                request_id=generate_request_id(), output_path=str(dest),
                output_file_name="video.mp4", save_output=True,
                perf_dump_path=str(dest / "perf.json"),
            )
            params.update(cell.get("request_overrides", {}))  # e.g. I2V: task + keyframe conditions
            # Explicit untimed warmup: copy of the identical request (default), or of the
            # previous cell's request (plan warmup='previous').
            warm_params = params
            if plan.get("warmup") == "previous" and prev_params is not None:
                warm_params = dict(prev_params, request_id=generate_request_id(), output_path=str(dest / "_warm_prev"), perf_dump_path=str(dest / "_warm_prev" / "perf.json"))
            prev_params = dict(params)
            sp = SamplingParams.from_user_sampling_params_args(
                server.model_path, server_args=server, **warm_params
            )
            req = prepare_request(server_args=server, sampling_params=sp)
            sp.prepare_video_request_for_queue(req)
            warm_start = time.perf_counter()
            # plan warmup='none': skip the per-cell warm-up; cells marked burnin still warm up.
            skip_warm = plan.get("warmup") == "none" and not cell.get("burnin")
            try:
                if skip_warm:
                    raise _SkipWarm
                warm = req.copy_as_warmup(warmup_steps)
                warm.extra["return_warmup_result"] = True
                warm.request_id = generate_request_id()
                if warm.metrics is not None:
                    warm.metrics.request_id = warm.request_id
                output = generator._send_to_scheduler_and_wait_for_response([warm])
                assert not getattr(output, "error", None), getattr(output, "error", None)
            except _SkipWarm:
                pass
            finally:
                sp.cleanup_video_request(req)
            warm_s = time.perf_counter() - warm_start
            effective = dict(data, prompt=cell["prompt"], seed=int(cell["seed"]), **cell.get("request_overrides", {}))
            (dest / "config.json").write_text(json.dumps(effective, indent=1) + "\n")
            (dest / "cell.json").write_text(json.dumps(dict(cell, arm=plan["arm"], index=index,
                operator=os.environ.get("T2T_OPERATOR_DIR", ""), warmup_steps=warmup_steps,
                explicit_warmup_wall_s=warm_s, session_startup_s=startup_s,
                session_plan=str(Path(opts.plan).resolve())), indent=1) + "\n")
            start = time.perf_counter()
            result = generator.generate(sampling_params_kwargs=params)
            elapsed = time.perf_counter() - start
            assert result is not None, "Generation failed"
            maybe_dump_performance(
                SimpleNamespace(perf_dump_path=str(dest / "perf.json")), server, cell["prompt"], result
            )
            assert (dest / "video.mp4").stat().st_size > 0 and (dest / "perf.json").exists()
            # the framework may dump perf.json a second time concurrently, leaving trailing data:
            # take the first complete object
            for attempt in range(40):  # the framework may still be rewriting it (truncate + write)
                try:
                    perf = json.JSONDecoder().raw_decode((dest / "perf.json").read_text())[0]
                    break
                except ValueError:
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"unreadable perf.json in {dest}")
            assert len(perf["denoise_steps_ms"]) >= int(data["num_inference_steps"]) - 1, "Missing denoise step metrics"
            (dest / "time.txt").write_text(f"client_wall_s={elapsed:.6f}\nwarmup_wall_s={warm_s:.6f}\n")
            (dest / "DONE").write_text("rc=0\n")
            print("CELL_DONE", cell["name"], round(warm_s, 1), round(elapsed, 1),
                  round(perf["total_duration_ms"] / 1000, 2), flush=True)
    finally:
        generator.shutdown()
    print("SESSION_DONE", round(time.perf_counter() - session_start, 1), flush=True)


if __name__ == "__main__":
    main()
