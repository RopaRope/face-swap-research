"""Face Swap Research: one-stop launcher (this is what the Windows .exe runs).

First run walks through everything:
  1. research-use notice (consenting people only; every frame is labelled as synthetic),
  2. downloads the model weights from their official sources and checks their SHA-256,
  3. checks the OBS virtual camera and explains how to set it up if it's missing,
  4. asks for the photo of the face to swap in and which webcam to use,
then starts the swap. Later runs remember the answers; run with --setup to change them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

import paths

APP = "Face Swap Research"
SETTINGS = paths.DATA / "settings.json"
OBS_URL = "https://obsproject.com/download"

# (file name, official URL, size in bytes, SHA-256)
MODELS = [
    ("inswapper_128.onnx",
     "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
     554253681, "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"),
    ("selfie_multiclass_256x256.tflite",
     "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
     "float32/latest/selfie_multiclass_256x256.tflite",
     16371837, "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
]
BUFFALO = ("buffalo_l.zip", "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
           288621354, "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f")
BUFFALO_FILES = ("det_10g.onnx", "w600k_r50.onnx")

NOTICE = f"""
==========================================================================
  {APP}
==========================================================================
  This tool replaces the face in your webcam video with another person's
  face, for deepfake research.

  * Use only faces of people who have agreed to take part.
  * Tell everyone on a call that the video is face-swapped.
  * Every frame carries the label "SYNTHETIC - AI FACE SWAP (research) /
    Face-swapped video - NOT the real person". It cannot be turned off.
  * The face models are licensed for non-commercial research only.
==========================================================================
"""


# ---------------------------------------------------------------- helpers

def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        ans = ""
    return ans.strip('"').strip("'") or default


def yes(prompt: str, default: bool = True) -> bool:
    ans = ask(f"{prompt} ({'Y/n' if default else 'y/N'})").lower()
    return default if not ans else ans.startswith("y")


def pause(msg: str = "Press Enter to continue..."):
    ask(msg)


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(s: dict):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(s, indent=2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, digest: str, tries: int = 3):
    """Download with progress and resume; verify size and SHA-256 before keeping the file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have < size:
                req = urllib.request.Request(url, headers={"User-Agent": "faceswap-research-setup"})
                if have:
                    req.add_header("Range", f"bytes={have}-")
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:          # server ignored the range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        t0, done = time.time(), have
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            done += len(chunk)
                            rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                            print(f"\r    {done / 1e6:7.1f} / {size / 1e6:.1f} MB  ({rate:.1f} MB/s)   ",
                                  end="", flush=True)
                print()
            print("    checking file...", end=" ", flush=True)
            if part.stat().st_size != size or sha256(part) != digest:
                print("damaged, retrying")
                part.unlink()
                continue
            print("ok")
            part.replace(dest)
            return
        except Exception as e:                     # network hiccup: keep the partial file, retry
            print(f"\n    download problem ({e}); retry {attempt}/{tries}")
            time.sleep(3)
    raise SystemExit(f"Could not download {dest.name} from {url}. Check the internet connection "
                     "and run the app again (it continues where it stopped).")


# ---------------------------------------------------------------- steps

def step_consent(s: dict):
    print(NOTICE)
    if s.get("agreed"):
        return
    if not yes("Only consenting people, and you'll tell call participants the video is swapped?",
               default=False):
        raise SystemExit("Setup stopped.")
    s["agreed"] = True
    save_settings(s)


def models_missing() -> list[str]:
    missing = [n for n, *_ in MODELS if not paths.model(n).exists()]
    buf = paths.INSIGHTFACE_ROOT / "models" / "buffalo_l"
    if not all((buf / f).exists() for f in BUFFALO_FILES):
        missing.append("buffalo_l")
    if not paths.model("parsing_bisenet.onnx").exists():
        missing.append("parsing_bisenet.onnx (should ship with the app)")
    return missing


def step_models():
    print("\n[1/3] Model weights")
    print(f"    folder: {paths.MODELS}")
    for name, url, size, digest in MODELS:
        if paths.model(name).exists():
            print(f"  - {name}: present")
            continue
        print(f"  - {name}: downloading {size / 1e6:.0f} MB from {url.split('/')[2]}")
        download(url, paths.MODELS / name, size, digest)

    buf = paths.INSIGHTFACE_ROOT / "models" / "buffalo_l"
    if all((buf / f).exists() for f in BUFFALO_FILES):
        print("  - buffalo_l (face detector + identity): present")
    else:
        name, url, size, digest = BUFFALO
        print(f"  - buffalo_l (face detector + identity): downloading {size / 1e6:.0f} MB from github.com")
        z = paths.INSIGHTFACE_ROOT / "models" / name
        download(url, z, size, digest)
        buf.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                base = os.path.basename(m)
                if base.endswith(".onnx"):         # flat copy; ignore folders inside the zip
                    with zf.open(m) as src, open(buf / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        z.unlink()
    left = models_missing()
    if left:
        raise SystemExit(f"Still missing: {', '.join(left)}")
    print("    all models ready.")


def vcam_ok() -> tuple[bool, str]:
    try:
        import pyvirtualcam
        with pyvirtualcam.Camera(width=640, height=360, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            return True, cam.device
    except Exception as e:
        return False, str(e)


def step_obs():
    print("\n[2/3] Virtual camera for video calls (OBS Studio)")
    while True:
        ok, info = vcam_ok()
        if ok:
            print(f"    found: {info}. In Zoom / Meet / Teams, choose it as your camera.")
            return True
        print(f"    not found ({info}).\n")
        print("    One-time setup:")
        print("      1. Install OBS Studio 30 or newer (free) from obsproject.com.")
        print("      2. Open OBS. If a setup wizard appears, you can cancel it.")
        print("      3. Click 'Start Virtual Camera' (bottom right, under Controls).")
        print("         If it warns that there are no sources, click Yes.")
        if sys.platform == "darwin":
            print("         macOS: allow the OBS camera extension in System Settings > General >")
            print("         Login Items & Extensions, then click 'Start Virtual Camera' again.")
        print("      4. Click 'Stop Virtual Camera' and quit OBS (this app sends the video itself).")
        print("      5. Restart your call app so it sees the new camera.")
        if yes("\n    Open the OBS download page in your browser?"):
            webbrowser.open(OBS_URL)
        choice = ask("    Press Enter to check again, or type 'skip' to run without a call camera").lower()
        if choice == "skip":
            return False


def step_source(s: dict):
    print("\n[3/3] Face and camera")
    import cv2

    cur = s.get("source")
    while True:
        hint = " (drag the photo onto this window, then press Enter)" if sys.platform == "win32" else ""
        p = ask(f"    Photo of the face to swap in{hint}", cur or "")
        img = cv2.imread(p) if p else None
        if img is None:
            print("    Could not read that image. Use a clear, front-facing photo (jpg/png).")
            continue
        dst = paths.DATA / "faces" / ("source" + (Path(p).suffix.lower() or ".jpg"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(p).resolve() != dst.resolve():
            shutil.copyfile(p, dst)
        s["source"] = str(dst)
        break

    import swap_live as sl
    names = sl._camera_names()
    real = [i for i, n in enumerate(names) if not any(w in n.lower() for w in sl.VIRTUAL_CAMERA_WORDS)]
    if names:
        print("    Cameras:")
        for i, n in enumerate(names):
            print(f"      {i}: {n}{'' if i in real else '   (virtual - not usable as input)'}")
    default = s.get("camera", real[0] if real else 0)
    cam = ask("    Camera number to use", str(default))
    s["camera"] = int(cam) if cam.isdigit() else default
    save_settings(s)


def run(s: dict, vcam: bool):
    import swap_live as sl

    argv = ["--source", s["source"], "--camera", str(s["camera"]), "--aligned", "--enhance", "0"]
    if vcam:
        argv.append("--vcam")
    print("\nStarting. A preview window opens; press q in it to stop.")
    if vcam:
        print("In your call app, pick the OBS Virtual Camera.")
    sl.main(argv)


def main():
    s = load_settings()
    force = "--setup" in sys.argv
    if force:
        s.pop("source", None)
    try:
        step_consent(s)
        step_models()
        vcam = step_obs()
        if force or not s.get("source") or not Path(s["source"]).exists() or "camera" not in s:
            step_source(s)
        else:
            print(f"\nUsing face photo {s['source']} and camera {s['camera']} (run with --setup to change).")
        run(s, vcam)
    except SystemExit as e:
        if e.code in (None, 0):
            raise
        print(f"\n{e.code}")
        if paths.FROZEN:
            pause("\nPress Enter to close.")
        raise SystemExit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        if paths.FROZEN:
            pause("\nSomething went wrong (details above). Press Enter to close.")
        raise


if __name__ == "__main__":
    main()
