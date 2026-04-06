from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .physics import DoubleParams, ModelName, SingleParams
from .render import RenderConfig, render_frames, render_to_gif, render_to_video
from .simulate import simulate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pendulums", description="Simulate and render single/double pendulums.")
    p.add_argument("--model", choices=["single", "double"], default="double")
    p.add_argument("--out", default="out.mp4", help="Output path (.mp4/.avi/.webm or .gif)")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--size", type=int, default=1080)
    p.add_argument("--trail", type=float, default=2.0, help="Trail length in seconds (0 disables)")

    # Integration knobs
    p.add_argument("--method", default="DOP853")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-11)
    p.add_argument("--max-step", type=float, default=0.0, help="0 means solver default")

    # Presets
    p.add_argument("--preset", choices=["calm", "chaotic"], default=None)

    # Single params
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--l", type=float, default=1.0)
    p.add_argument("--g", type=float, default=9.81)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=0.0)

    # Double params
    p.add_argument("--m1", type=float, default=1.0)
    p.add_argument("--m2", type=float, default=1.0)
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--theta1", type=float, default=1.2)
    p.add_argument("--omega1", type=float, default=0.0)
    p.add_argument("--theta2", type=float, default=1.0)
    p.add_argument("--omega2", type=float, default=0.0)

    return p


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return

    if args.model == "single":
        if args.preset == "calm":
            args.theta = 0.6
            args.omega = 0.0
        elif args.preset == "chaotic":
            args.theta = 2.2
            args.omega = 0.0
        return

    # double
    if args.preset == "calm":
        args.theta1 = 0.9
        args.theta2 = 0.8
        args.omega1 = 0.0
        args.omega2 = 0.0
    elif args.preset == "chaotic":
        args.theta1 = 2.1
        args.theta2 = 1.0
        args.omega1 = 0.0
        args.omega2 = 0.0


def _to_params_and_y0(args: argparse.Namespace) -> tuple[ModelName, Any, np.ndarray]:
    model: ModelName = args.model
    if model == "single":
        params = SingleParams(m=args.m, l=args.l, g=args.g)
        y0 = np.array([args.theta, args.omega], dtype=float)
        return model, params, y0

    params = DoubleParams(m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2, g=args.g)
    y0 = np.array([args.theta1, args.omega1, args.theta2, args.omega2], dtype=float)
    return model, params, y0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_preset(args)

    model, params, y0 = _to_params_and_y0(args)

    res = simulate(
        model=model,
        params=params,
        y0=y0,
        duration=args.duration,
        fps=args.fps,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        max_step=None if args.max_step <= 0 else args.max_step,
    )

    cfg = RenderConfig(size=args.size, fps=args.fps, trail_seconds=args.trail)
    frames = render_frames(model=model, params=params, t=res.t, y=res.y, energies=res.energies, cfg=cfg)

    out = Path(args.out)
    if out.suffix.lower() == ".gif":
        render_to_gif(frames=frames, out_path=out, fps=args.fps)
    else:
        render_to_video(frames=frames, out_path=out, fps=args.fps)

    info = {
        "out": str(out),
        "model": model,
        "params": asdict(params),
        "duration": float(args.duration),
        "fps": float(args.fps),
        "frames": int(len(frames)),
        "energy_drift_abs_max": float(res.energy_drift_abs_max),
        "energy_drift_rel_max": float(res.energy_drift_rel_max),
    }
    print(info)
    return 0

