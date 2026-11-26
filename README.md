# Live face swap — research prototype

Webcam → SCRFD detection → ArcFace identity embedding → inswapper_128 swap → paste-back → display.
Tuned for Apple Silicon (ONNX Runtime CoreML provider, CPU fallback).

**Research use only.** Use faces of people who have consented. Every output frame carries a
visible `SYNTHETIC` label. The pretrained weights are licensed for non-commercial research only.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Models:
- `buffalo_l` (detector + embedder): downloaded automatically by insightface to `~/.insightface/models/`
  from the official deepinsight/insightface GitHub releases on first run.
- `inswapper_128.onnx` (~530 MB): place at `models/inswapper_128.onnx`.

## Run

```bash
.venv/bin/python swap_live.py --source faces/me.jpg
```

Keys: `space` toggles the swap on/off, `q`/`esc` quits. Speed knobs: `--det-size 256`,
`--detect-every 3`, `--width 960 --height 540`. Landmark smoothing: `--smooth 0.5` (lower = steadier, laggier).

### Frame-aligned mode (no colour lag)

```bash
.venv/bin/python swap_live.py --source faces/me.jpg --aligned
```

Each frame waits for the skin mask computed on *that* frame (2 segmenters run in parallel), so
recolouring never trails behind movement. Cost: output is ~0.27 s behind the camera. On an M1 with
one face: 6.8 fps, 270 ms delay (`--max-inflight 2`: 7.7 fps, 380 ms). Camera frames that arrive
while the pipeline is busy are dropped, so the delay never grows.

### Keyframe mode (smooth on slow machines)

```bash
.venv/bin/python swap_live.py --source faces/me.jpg --keyframes
```

The full swap (skin mask, recolour, inswapper) runs on "keyframes" as often as the machine
manages; every camera frame is still shown, by re-detecting the face landmarks and warping the
nearest swapped face onto them. The output runs `--delay` (default 0.35 s) behind the camera so
the nearest keyframe can be slightly *after* the frame being shown. On an M1 (one face): 28 fps
shown, 4.5 full swaps/s, ~0.4 s delay. Warped frames differ from a true full swap by ~5/255 on
average. Head motion is smooth; mouth/eye changes update at the keyframe rate. Only the largest
face is swapped. `--max-keyframe-rate N` caps full swaps per second (frees CPU for other apps).
Combine with `--vcam` for calls.

### Virtual camera (for test calls between consenting participants)

```bash
.venv/bin/python swap_live.py --source faces/me.jpg --keyframes --vcam
```

Sends the swapped video to a virtual camera you can pick in Zoom / Meet / Teams. One-time setup on macOS: install OBS Studio
30+, click **Start Virtual Camera** once, approve the system extension in System Settings, quit
OBS. Your voice is not delayed, so it runs ~0.27 s ahead of your lips.

## Windows app

`faceswap_app.py` is a guided launcher, and the Windows `.exe` runs it. On first start it:
1. shows the research-use notice;
2. downloads the weights from their official sources into `%LOCALAPPDATA%\FaceSwapResearch` and
   checks each one's SHA-256 (~860 MB; an interrupted download resumes);
3. checks for the OBS virtual camera and walks you through installing it;
4. asks for the face photo and the webcam.

It then runs the frame-aligned swap into the virtual camera. Later starts skip straight to the
swap; `FaceSwapResearch.exe --setup` changes the photo and camera. It uses no PyTorch: the face
parser runs as ONNX (`models/parsing_bisenet.onnx`, shipped inside the app, made with
`python face_parse.py`). The GPU goes through DirectML, which works with Intel integrated graphics
too. The GFPGAN enhancer is not included.

Build it **on a Windows PC** (PyInstaller can't cross-compile from macOS):
1. Install Python 3.12 (64-bit, python.org) and Visual Studio Build Tools with "Desktop
   development with C++". insightface compiles a small C++ part when it installs.
2. Copy this project folder over, including `models/parsing_bisenet.onnx`. The other weights
   aren't needed because the app downloads them.
3. Run `windows\build.bat`. The result is `dist\FaceSwapResearch\FaceSwapResearch.exe`, plus a zip
   of that folder to copy to other PCs.

Or push the project to GitHub and run the `windows-exe` workflow (Actions tab); it builds the same
folder on GitHub's Windows machines.

Still image (headless test):

```bash
.venv/bin/python swap_live.py --source faces/me.jpg --image target.jpg --out out.jpg
```

## Known install fix

`basicsr` (pulled in by `gfpgan`) imports `torchvision.transforms.functional_tensor`, which newer
torchvision removed. After `pip install`, run:

```bash
sed -i '' 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/' .venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

`basicsr` also imports `distutils` (removed in Python 3.12):

```bash
sed -i '' 's/^from distutils.version import LooseVersion$/from packaging.version import Version as LooseVersion/' .venv/lib/python3.12/site-packages/basicsr/archs/arch_util.py
```
