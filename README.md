# simulation of rotating bodies

This repo now includes a small, reproducible Python package `pendulums` (single + double pendulum) with:
- **SciPy** integration (`solve_ivp`, energy drift monitoring)
- **OpenCV** rendering to **MP4/WebM/AVI**
- optional **GIF** output via `imageio` + `pillow`

## Quick start (uv)

Demo output:

![](out.gif)

Install dependencies:

```bash
uv sync
```

Optional GIF dependencies:

```bash
uv sync --extra gif
```

Run double pendulum (MP4):

```bash
uv run pendulums --model double --preset chaotic --duration 10 --fps 60 --size 1080 --out out.mp4
```

Run single pendulum:

```bash
uv run pendulums --model single --preset calm --duration 10 --fps 60 --size 1080 --out single.mp4
```

GIF (requires `--extra gif`):

```bash
uv run pendulums --model double --preset chaotic --duration 8 --fps 60 --size 720 --out out.gif
```

## Notes
- If `.webm` export fails, your OpenCV build may lack the needed codec; `.mp4` is the default recommendation.
- Existing legacy demo files (`main.py`, `make_video.py`, etc.) are kept as-is.
