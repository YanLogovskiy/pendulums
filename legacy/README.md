## Legacy demo (matplotlib + RK4)

This folder contains the original demo code (linear oscillator + simple pendulum) with:
- rendering via **matplotlib** (frames → video)
- numerical integration via a simple **RK4**

### Run

From the repository root:

```bash
cd legacy
python3 main.py
```

### Notes
- The demo writes frames into `frames/` and videos into `videos/`.
- `ffmpeg` is used to assemble the video (`.avi`).
- You may need these system/python deps installed:
  - `python3`, `matplotlib`, `numpy`
  - `ffmpeg`

