from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .physics import DoubleParams, ModelName, Params, SingleParams, positions


@dataclass(frozen=True, slots=True)
class RenderConfig:
    size: int = 1080  # square frame: size x size
    fps: float = 60.0
    trail_seconds: float = 2.0
    margin_px: int = 80
    rod_thickness: int = 6
    bob_radius: int = 18
    trail_thickness: int = 2
    show_overlay: bool = True


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required. Install with: uv sync") from e
    return cv2


def _world_to_px(p: np.ndarray, *, center_px: np.ndarray, scale: float) -> tuple[int, int]:
    # world: x right, y up; image: x right, y down
    x, y = float(p[0]), float(p[1])
    px = int(round(center_px[0] + scale * x))
    py = int(round(center_px[1] - scale * y))
    return px, py


def _world_extent(model: ModelName, params: Params) -> float:
    if model == "single":
        assert isinstance(params, SingleParams)
        return params.l
    assert isinstance(params, DoubleParams)
    return params.l1 + params.l2


def render_frames(
    *,
    model: ModelName,
    params: Params,
    t: np.ndarray,
    y: np.ndarray,
    energies: Optional[np.ndarray] = None,
    cfg: RenderConfig = RenderConfig(),
) -> list[np.ndarray]:
    cv2 = _require_cv2()

    size = int(cfg.size)
    if size <= 0:
        raise ValueError("cfg.size must be > 0")
    if y.ndim != 2:
        raise ValueError("y must have shape (state_dim, N)")
    n = y.shape[1]
    if t.shape != (n,):
        raise ValueError("t must have shape (N,)")
    if energies is not None and energies.shape != (n,):
        raise ValueError("energies must have shape (N,)")

    center = np.array([size // 2, size // 2], dtype=float)

    extent = _world_extent(model, params)
    usable = max(1, size - 2 * int(cfg.margin_px))
    scale = usable / (2.0 * extent)  # world range approx [-extent, +extent]

    # Colors (BGR)
    bg = (10, 10, 14)
    rod = (230, 230, 230)
    bob1 = (80, 190, 255)
    bob2 = (255, 140, 80)
    trail = (120, 120, 255)
    text = (245, 245, 245)
    shadow = (0, 0, 0)

    trail_len = int(max(0.0, cfg.trail_seconds) * cfg.fps)
    trail_pts: list[tuple[int, int]] = []

    frames: list[np.ndarray] = []
    e0 = float(energies[0]) if energies is not None else 0.0

    pivot = (int(center[0]), int(center[1]))

    for i in range(n):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = bg

        pos = positions(model, y[:, i], params)
        if model == "single":
            p1 = pos[0]
            p1_px = _world_to_px(p1, center_px=center, scale=scale)

            cv2.line(frame, pivot, p1_px, rod, thickness=cfg.rod_thickness, lineType=cv2.LINE_AA)
            cv2.circle(frame, p1_px, cfg.bob_radius, bob1, thickness=-1, lineType=cv2.LINE_AA)

            trail_pts.append(p1_px)
        else:
            p1 = pos[0]
            p2 = pos[1]
            p1_px = _world_to_px(p1, center_px=center, scale=scale)
            p2_px = _world_to_px(p2, center_px=center, scale=scale)

            cv2.line(frame, pivot, p1_px, rod, thickness=cfg.rod_thickness, lineType=cv2.LINE_AA)
            cv2.line(frame, p1_px, p2_px, rod, thickness=cfg.rod_thickness, lineType=cv2.LINE_AA)

            cv2.circle(frame, p1_px, cfg.bob_radius, bob1, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, p2_px, cfg.bob_radius, bob2, thickness=-1, lineType=cv2.LINE_AA)

            trail_pts.append(p2_px)

        if trail_len > 1:
            trail_pts = trail_pts[-trail_len:]
            if len(trail_pts) >= 2:
                pts = np.array(trail_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame, [pts], isClosed=False, color=trail, thickness=cfg.trail_thickness, lineType=cv2.LINE_AA
                )

        if cfg.show_overlay:
            overlay_lines = [f"t = {t[i]:.2f}s", f"model = {model}"]
            if energies is not None:
                drel = (float(energies[i]) - e0) / (abs(e0) + 1e-12)
                overlay_lines.append(f"dE/E0 = {drel:+.2e}")

            x0, y0 = 24, 44
            for j, line in enumerate(overlay_lines):
                y_text = y0 + j * 32
                cv2.putText(
                    frame,
                    line,
                    (x0 + 2, y_text + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    shadow,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (x0, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    text,
                    2,
                    cv2.LINE_AA,
                )

        frames.append(frame)

    return frames


def render_to_video(
    *,
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: float,
) -> Path:
    cv2 = _require_cv2()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("frames is empty")
    h, w = frames[0].shape[:2]

    ext = out_path.suffix.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif ext == ".webm":
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
    else:
        raise ValueError("Unsupported video extension. Use .mp4, .avi, or .webm")

    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path} (codec may be unavailable)")

    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                raise ValueError("All frames must have same size")
            writer.write(f)
    finally:
        writer.release()

    return out_path


def render_to_gif(
    *,
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: float,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GIF output requires: uv sync --extra gif") from e

    if not frames:
        raise ValueError("frames is empty")

    rgb_frames = [f[:, :, ::-1] for f in frames]  # BGR -> RGB
    duration = 1.0 / float(fps)
    imageio.mimsave(out_path, rgb_frames, duration=duration)
    return out_path

