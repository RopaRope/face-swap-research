"""Live face swap research prototype (webcam -> detect -> embed -> swap -> display).

Pipeline:
  SCRFD detector + ArcFace embedder  (insightface buffalo_l pack)
  inswapper_128 one-shot swap model  (models/inswapper_128.onnx)

Every output frame carries a visible "SYNTHETIC" label. Use only faces of
people who have consented to this research.

Usage:
  python swap_live.py --source faces/me.jpg                    # live webcam
  python swap_live.py --source faces/me.jpg --aligned          # no colour lag (adds ~0.2 s delay)
  python swap_live.py --source faces/me.jpg --aligned --vcam   # also feed a virtual camera
  python swap_live.py --source faces/me.jpg --image in.jpg     # single still -> out.jpg
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

import paths
from skin_tone import SkinToneMatcher

ROOT = paths.ROOT
DEFAULT_SWAPPER = paths.model("inswapper_128.onnx")
LABEL = ("")


def providers() -> list:
    avail = ort.get_available_providers()
    # CoreML on Apple Silicon; DirectML on Windows (any DirectX 12 GPU, Intel iGPUs included).
    return [p for p in ("CoreMLExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
            if p in avail]


def load_models(swapper_path: Path, det_size: int, fp16: bool = True):
    app = FaceAnalysis(name="buffalo_l", root=str(paths.INSIGHTFACE_ROOT),
                       allowed_modules=["detection", "recognition"], providers=providers())
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    if not swapper_path.exists():
        sys.exit(f"Swap model not found: {swapper_path}\nRun the setup (faceswap_app) to download it.")
    if fp16 and "CoreMLExecutionProvider" in providers():
        # fp16 + CoreML MLProgram is ~1.7x faster on Apple Silicon (100 -> 59 ms/face on M1),
        # with output differing from fp32 by <0.1/255 on average.
        path16 = swapper_path.with_name(swapper_path.stem + "_fp16_local.onnx")
        if not path16.exists():
            import onnx
            from onnxconverter_common import float16
            print("Building fp16 swap model (one-time)...")
            onnx.save(float16.convert_float_to_float16(onnx.load(str(swapper_path)), keep_io_types=True),
                      str(path16))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(path16), so, providers=[
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
            "CPUExecutionProvider"])
        return app, INSwapper(str(path16), session=sess)
    swapper = insightface.model_zoo.get_model(str(swapper_path), providers=providers())
    return app, swapper


def load_parser():
    """BiSeNet face parser. PyTorch on an Apple GPU (31 ms vs 58 ms for ONNX on the CPU); otherwise
    the exported ONNX model, which needs no PyTorch (the Windows app)."""
    onnx_path = paths.model("parsing_bisenet.onnx")
    try:
        import torch
        mps = torch.backends.mps.is_available()
    except ImportError:
        mps = False
    if onnx_path.exists() and not mps:
        from face_parse import OnnxFaceParser
        return OnnxFaceParser(onnx_path, [p for p in providers() if p != "CoreMLExecutionProvider"])
    from enhance import FaceParser
    return FaceParser()


def source_face(app: FaceAnalysis, path: str):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read source image: {path}")
    faces = app.get(img)
    if not faces:
        sys.exit(f"No face found in source image: {path}")
    # Largest face in the source photo is the identity to transfer.
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def watermark(frame: np.ndarray) -> np.ndarray:
    """Burn the disclosure label into the top-left corner. Applied to every output, never optional."""
    h, w = frame.shape[:2]
    scale = max(0.55, w / 1280 * 0.8)
    thick = max(1, int(2 * scale))
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for t in LABEL]
    tw = max(s[0] for s in sizes)
    th = sizes[0][1]
    gap = int(th * 0.7)
    x, y0 = 12, 12
    y1 = y0 + len(LABEL) * (th + gap) + gap
    cv2.rectangle(frame, (x - 6, y0 - 4), (x + tw + 8, y1), (0, 0, 0), -1)
    for i, t in enumerate(LABEL):
        y = y0 + gap + th + i * (th + gap)
        color = (0, 0, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return frame


class KpsSmoother:
    """Exponential moving average on 5-point landmarks to cut frame-to-frame jitter."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev: list[np.ndarray] = []

    def __call__(self, faces):
        faces = sorted(faces, key=lambda f: f.bbox[0])
        if len(faces) != len(self.prev) or self.alpha >= 1.0:
            self.prev = [f.kps.copy() for f in faces]
            return faces
        for i, f in enumerate(faces):
            # Reset if the face jumped (new person / big motion).
            if np.abs(f.kps - self.prev[i]).mean() > 25:
                self.prev[i] = f.kps.copy()
            else:
                self.prev[i] = self.alpha * f.kps + (1 - self.alpha) * self.prev[i]
            f.kps = self.prev[i].astype(np.float32)
        return faces


def detect(app: FaceAnalysis, frame: np.ndarray):
    """Detection + 5-point landmarks only; target faces don't need an identity embedding."""
    bboxes, kpss = app.det_model.detect(frame, max_num=0, metric="default")
    return [Face(bbox=b[:4], kps=k, det_score=b[4]) for b, k in zip(bboxes, kpss)]


def swap_frame(frame, app, swapper, src, faces=None, skin=None, enhancer=None, body=None):
    if faces is None:
        faces = detect(app, frame)
    # Recolour the target's skin to the source tone first, so face and body match after the swap.
    out = skin.apply(frame, faces, body=body) if skin is not None else frame.copy()
    if out is frame:
        out = frame.copy()
    H, W = out.shape[:2]
    for f in faces:
        # Blend only inside a padded box around the face: paste-back warps/blurs
        # whole-image masks, which costs far more at 720p than the face itself.
        x1, y1, x2, y2 = f.bbox
        pad = 0.6 * max(x2 - x1, y2 - y1)
        rx1, ry1 = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
        rx2, ry2 = min(int(x2 + pad), W), min(int(y2 + pad), H)
        roi = out[ry1:ry2, rx1:rx2]
        rf = Face(bbox=f.bbox - [rx1, ry1, rx1, ry1], kps=f.kps - [rx1, ry1], det_score=f.det_score)
        out[ry1:ry2, rx1:rx2] = swapper.get(roi, rf, src, paste_back=True)
    if enhancer is not None:
        out = enhancer.enhance(out, faces)
    return out, faces


def run_image(args, app, swapper, src, skin=None, enhancer=None):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read image: {args.image}")
    t = time.perf_counter()
    out, faces = swap_frame(img, app, swapper, src, skin=skin, enhancer=enhancer)
    dt = (time.perf_counter() - t) * 1000
    cv2.imwrite(args.out, watermark(out))
    print(f"Swapped {len(faces)} face(s) in {dt:.0f} ms -> {args.out}")


VIRTUAL_CAMERA_WORDS = ("virtual", "obs", "snap camera", "mmhmm", "ndi", "camo")


def _camera_backend():
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW      # DirectShow: indices follow the order pygrabber lists names in
    return cv2.CAP_ANY


def _camera_names() -> list[str]:
    """Camera names in OpenCV's index order for this platform's backend; [] if unknown.
    macOS: OpenCV numbers cameras by sorting their unique IDs, so sort system_profiler's list the
    same way. Windows: DirectShow device order, via pygrabber."""
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(FilterGraph().get_input_devices())
        except Exception:
            return []
    if sys.platform != "darwin":
        return []
    import json
    import subprocess
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True,
                             text=True, timeout=10).stdout
        cams = json.loads(out).get("SPCameraDataType", [])
        cams.sort(key=lambda c: c.get("spcamera_unique-id", ""))
        return [c.get("_name", "") for c in cams]
    except Exception:
        return []


def _pick_camera(args) -> int:
    """--camera if given; else the first real camera, skipping virtual ones (OBS Virtual Camera
    lists itself first once installed, and reading it would feed the output back to the input)."""
    if args.camera >= 0:
        return args.camera
    names = _camera_names()
    for i, name in enumerate(names):
        if not any(wd in name.lower() for wd in VIRTUAL_CAMERA_WORDS):
            return i
    return 0


def open_camera(args):
    backend = _camera_backend()
    args.camera = _pick_camera(args)
    names = _camera_names()
    cap = cv2.VideoCapture(args.camera, backend)
    if sys.platform == "win32":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # DirectShow: 30 fps at 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Could not open camera. " + (
            "Grant camera access to your terminal app (System Settings > Privacy & Security > Camera)."
            if sys.platform == "darwin" else
            "Check Windows Settings > Privacy & security > Camera allows desktop apps, and close other "
            "apps using the camera."))

    # macOS cameras often return empty frames for a moment while warming up.
    deadline = time.perf_counter() + 5
    ok = False
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)
    if not ok:
        sys.exit(f"Camera {args.camera} opened but sent no frames for 5 s. Close other apps using "
                 "the camera, or try --camera 1.")
    name = f" ({names[args.camera]})" if args.camera < len(names) else ""
    print(f"Camera {args.camera}{name}: {frame.shape[1]}x{frame.shape[0]}")
    return cap, frame


OBS_HELP = (
    "Install OBS Studio 30 or newer from https://obsproject.com, open it, click 'Start Virtual Camera' "
    "once (answer Yes if it warns about no sources), then 'Stop Virtual Camera' and quit OBS."
    + (" On macOS also approve the OBS camera extension in System Settings > General > "
       "Login Items & Extensions." if sys.platform == "darwin" else ""))


def open_vcam(w: int, h: int, fps: int):
    import pyvirtualcam
    try:
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
    except RuntimeError as e:
        sys.exit(f"Virtual camera unavailable: {e}\n{OBS_HELP}")
    print(f"Virtual camera: {cam.device} ({w}x{h}). Pick it as the camera in your call app.")
    return cam


class AlignedFrames:
    """Webcam frames paired with a skin mask computed on that same frame.

    A capture thread reads the camera; each kept frame goes to a pool of segmentation workers
    (one MediaPipe segmenter each). The consumer gets frames in capture order, each with its own
    mask, so recolouring never uses a stale mask (no colour trailing behind motion). At most
    `max_inflight` frames are queued; extra camera frames are dropped, so delay stays bounded
    instead of growing when processing is slower than the camera.
    """

    def __init__(self, cap, workers: int, max_inflight: int, need_mask: bool):
        self.cap = cap
        self.q: queue.Queue = queue.Queue(maxsize=max_inflight)
        self.stop = threading.Event()
        self.pool = None
        if need_mask and workers > 0:
            from skin_tone import SkinSegmenter
            self._segs: queue.Queue = queue.Queue()
            for _ in range(workers):
                self._segs.put(SkinSegmenter())
            self.pool = ThreadPoolExecutor(workers)
        self.dropped = 0
        threading.Thread(target=self._capture, daemon=True).start()

    def _segment(self, frame):
        seg = self._segs.get()
        try:
            return seg.skin_prob(frame)
        finally:
            self._segs.put(seg)

    def _read(self):
        """(capture time, frame) or None. `cap` is a cv2.VideoCapture or a FrameBus (read_stamped)."""
        if hasattr(self.cap, "read_stamped"):
            return self.cap.read_stamped()
        ok, frame = self.cap.read()
        return (time.perf_counter(), frame) if ok and frame is not None else None

    def _capture(self):
        while not self.stop.is_set():
            item = self._read()
            if item is None:
                time.sleep(0.01)
                continue
            if self.q.full():
                self.dropped += 1
                continue
            t, frame = item
            try:
                fut = self.pool.submit(self._segment, frame) if self.pool else None
            except RuntimeError:          # pool shut down: we're closing
                return
            self.q.put((t, frame, fut))

    def get(self, timeout: float = 3.0):
        """(capture time, frame, mask or None). Blocks until that frame's mask is ready."""
        t, frame, fut = self.q.get(timeout=timeout)
        return t, frame, (fut.result() if fut else None)

    def close(self):
        self.stop.set()
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)


def run_aligned(args, app, swapper, src, skin=None, enhancer=None):
    cap, first = open_camera(args)
    h, w = first.shape[:2]
    vcam = open_vcam(w, h, args.vcam_fps) if args.vcam else None
    frames = AlignedFrames(cap, args.seg_workers, args.max_inflight, need_mask=skin is not None)

    win = "Face swap (research)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

    faces, frame_i, fps, lat, last = [], 0, 0.0, 0.0, time.perf_counter()
    enabled, skin_on, enh_on = True, True, args.enhance_live
    print("Running (frame-aligned). Keys: [space] swap, [k] skin tone, [e] enhancer, [q]/[esc] quit.")
    try:
        while True:
            try:
                t_cap, frame, body = frames.get()
            except queue.Empty:
                print("Camera stopped sending frames.")
                break
            # Process the unmirrored frame: call apps mirror the self-view themselves, and the other
            # side should see the normal (unmirrored) image with readable label text.
            if enabled:
                # Detect on every frame, unsmoothed: frames here are ~150 ms apart, so reusing the
                # previous detection or averaging with it puts the face swap and the skin colour
                # where the face *was*, which shows as colour trailing behind movement.
                faces = detect(app, frame)
                out, _ = swap_frame(frame, app, swapper, src, faces, skin=skin if skin_on else None,
                                    enhancer=enhancer if enh_on else None, body=body)
            else:
                out = frame.copy()
            frame_i += 1

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            lat = 0.9 * lat + 0.1 * (now - t_cap) if frame_i > 1 else now - t_cap
            last = now

            if vcam is not None:
                vcam.send(watermark(out.copy()))

            preview = cv2.flip(out, 1) if args.mirror else out
            watermark(preview)
            cv2.imshow(win, preview)
            key = cv2.waitKey(1) & 0xFF
            if frame_i % 60 == 0:
                print(f"{fps:.1f} fps, delay {lat * 1000:.0f} ms, {len(faces)} face(s), "
                      f"dropped {frames.dropped} camera frames", flush=True)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                enabled = not enabled
            if key == ord("k"):
                skin_on = not skin_on
            if key == ord("e"):
                enh_on = not enh_on
    finally:
        frames.close()
        if vcam is not None:
            vcam.close()
        cap.release()
        cv2.destroyAllWindows()


class KeyframeSession:
    """Camera -> keyframe worker -> per-frame warp. Shared by the live loop and the benchmark."""

    def __init__(self, cap, app, swapper, src, skin, parser, args):
        from keyframe import FrameBus, KeyframeWorker
        self.app, self.skin, self.delay = app, skin, args.delay
        self.bus = FrameBus(cap)
        self.frames = AlignedFrames(self.bus, args.seg_workers, 1, need_mask=skin is not None)
        self.worker = KeyframeWorker(
            self.frames, lambda f: detect(app, f),
            lambda f, faces: swap_frame(f, app, swapper, src, faces)[0],
            skin, parser, max_rate=args.max_keyframe_rate)
        self.smooth = KpsSmoother(args.smooth)
        self.seq = -1
        self.skipped = 0

    def next(self):
        """Block until the next frame is `delay` old; return (capture time, output frame) or None."""
        from keyframe import render
        if self.seq < 0:
            item = self.bus.at(0)
            if item is None:
                return None
            self.seq = item[0]
        # Fallen behind (display slower than the camera)? Jump to the newest frame that is due.
        due = self.bus.seq_before(time.perf_counter() - self.delay)
        if due > self.seq + 1:
            self.skipped += due - self.seq
            self.seq = due
        item = self.bus.at(self.seq)
        if item is None:
            return None
        s, t, frame = item
        self.seq = s + 1
        wait = t + self.delay - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        faces = self.smooth(detect(self.app, frame)) if self.worker.enabled else []
        key = self.worker.nearest(t) if self.worker.enabled else None
        skin = self.skin if self.worker.skin_on else None
        return t, render(frame, t, faces, key, skin)

    def close(self):
        self.worker.close()
        self.frames.close()
        self.bus.close()


def run_keyframes(args, app, swapper, src, skin, parser):
    cap, first = open_camera(args)
    h, w = first.shape[:2]
    vcam = open_vcam(w, h, args.vcam_fps) if args.vcam else None
    sess = KeyframeSession(cap, app, swapper, src, skin, parser, args)

    win = "Face swap (research)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    print(f"Running (keyframes, {args.delay:.2f} s delay). Keys: [space] swap, [k] skin tone, [q]/[esc] quit.")
    n, fps, last, k_last, k_rate = 0, 0.0, time.perf_counter(), 0, 0.0
    try:
        while True:
            res = sess.next()
            if res is None:
                print("Camera stopped sending frames.")
                break
            t, out = res
            n += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            if n % 30 == 0:
                k_rate = (sess.worker.count - k_last) / max(30 / max(fps, 1e-3), 1e-3)
                k_last = sess.worker.count
            if vcam is not None:
                vcam.send(watermark(out.copy()))
            preview = cv2.flip(out, 1) if args.mirror else out
            watermark(preview)
            cv2.imshow(win, preview)
            key = cv2.waitKey(1) & 0xFF
            if n % 60 == 0:
                print(f"{fps:.1f} fps shown, {k_rate:.1f} full swaps/s, delay {(now - t) * 1000:.0f} ms, "
                      f"skipped {sess.skipped} frames", flush=True)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                sess.worker.enabled = not sess.worker.enabled
            if key == ord("k"):
                sess.worker.skin_on = not sess.worker.skin_on
    finally:
        sess.close()
        if vcam is not None:
            vcam.close()
        cap.release()
        cv2.destroyAllWindows()


def run_live(args, app, swapper, src, skin=None, enhancer=None):
    cap, frame = open_camera(args)

    win = "Face swap (research)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

    smooth = KpsSmoother(args.smooth)
    faces, frame_i, fps, last = [], 0, 0.0, time.perf_counter()
    enabled = True
    skin_on = True
    enh_on = args.enhance_live   # GFPGAN is ~300 ms/face on an M1: off by default live, [e] toggles
    misses = 0
    print("Running. Keys: [space] swap, [k] skin tone, [e] enhancer, [q]/[esc] quit.")
    stage = {"read": 0.0, "detect": 0.0, "swap": 0.0, "show": 0.0}
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        stage["read"] += time.perf_counter() - t0
        if not ok or frame is None:
            misses += 1
            if misses > 60:
                print("Camera stopped sending frames.")
                break
            cv2.waitKey(10)
            continue
        misses = 0
        if args.mirror:
            frame = cv2.flip(frame, 1)

        if enabled:
            t0 = time.perf_counter()
            if frame_i % args.detect_every == 0:
                faces = smooth(detect(app, frame))
            t1 = time.perf_counter()
            out, _ = swap_frame(frame, app, swapper, src, faces, skin=skin if skin_on else None,
                                enhancer=enhancer if enh_on else None)
            stage["detect"] += t1 - t0
            stage["swap"] += time.perf_counter() - t1
        else:
            out = frame.copy()
        t_show = time.perf_counter()
        frame_i += 1

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        watermark(out)
        cv2.imshow(win, out)
        if frame_i == 1:
            print("Window open (titled 'Face swap (research)').")
        key = cv2.waitKey(1) & 0xFF
        stage["show"] += time.perf_counter() - t_show
        if frame_i % 60 == 0:
            ms = "  ".join(f"{k} {v / 60 * 1000:.0f}ms" for k, v in stage.items())
            print(f"{fps:.1f} fps, {len(faces)} face(s)  |  {ms}", flush=True)
            stage = dict.fromkeys(stage, 0.0)
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            enabled = not enabled
        if key == ord("k"):
            skin_on = not skin_on
        if key == ord("e"):
            enh_on = not enh_on

    cap.release()
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="photo of the (consenting) identity to swap in")
    ap.add_argument("--swapper", default=str(DEFAULT_SWAPPER))
    ap.add_argument("--image", help="process one still image instead of the webcam")
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--camera", type=int, default=-1,
                    help="camera index (default: first real camera, skipping virtual cameras)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--det-size", type=int, default=320, help="detector input size (smaller = faster)")
    ap.add_argument("--detect-every", type=int, default=2, help="run detection every N frames")
    ap.add_argument("--smooth", type=float, default=0.6, help="landmark EMA alpha (1.0 = off)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--fp32", action="store_true", help="use the original fp32 swap model (slower on Mac)")
    ap.add_argument("--skin", type=float, default=1.0,
                    help="shift target's visible skin to the source tone (0 = off, 1 = full)")
    ap.add_argument("--enhance", type=float, default=1.0,
                    help="GFPGAN face restoration strength (0 = don't load, 1 = full)")
    ap.add_argument("--enhance-live", action="store_true",
                    help="start live mode with the enhancer on (slow on Mac; [e] toggles it anyway)")
    ap.add_argument("--aligned", action="store_true",
                    help="wait for each frame's own skin mask before recolouring: no colour lag, "
                         "output ~0.2 s behind the camera")
    ap.add_argument("--seg-workers", type=int, default=2, help="--aligned: parallel skin segmenters")
    ap.add_argument("--max-inflight", type=int, default=1,
                    help="--aligned: frames queued at most (bounds the delay; extra frames are dropped)")
    ap.add_argument("--vcam", action="store_true",
                    help="also send the labelled output to a virtual camera (implies --aligned; "
                         "macOS needs OBS's virtual camera installed)")
    ap.add_argument("--vcam-fps", type=int, default=30)
    ap.add_argument("--keyframes", action="store_true",
                    help="smooth mode: full swap on as many frames as the machine manages, every other "
                         "frame shown by warping the nearest swapped face onto its landmarks")
    ap.add_argument("--delay", type=float, default=0.35,
                    help="--keyframes: seconds the output runs behind the camera")
    ap.add_argument("--max-keyframe-rate", type=float, default=0.0,
                    help="--keyframes: cap full swaps per second (0 = as many as possible)")
    args = ap.parse_args(argv)
    if args.vcam and not args.keyframes:
        args.aligned = True
    if args.keyframes:
        args.aligned = True       # keyframes use frame-aligned skin masks too

    print("ONNX Runtime providers:", providers())
    app, swapper = load_models(Path(args.swapper), args.det_size, fp16=not args.fp32)
    src = source_face(app, args.source)
    parser = None
    if args.skin > 0 or args.enhance > 0 or args.keyframes:
        parser = load_parser()     # shared: same-frame face/neck skin mask + enhancer paste mask
    skin = None
    if args.skin > 0:
        skin = SkinToneMatcher(cv2.imread(args.source), src, strength=args.skin,
                               live=not args.image and not args.aligned, parser=parser)
    enhancer = None
    if args.enhance > 0:
        try:
            from enhance import FaceEnhancer
            enhancer = FaceEnhancer(weight=args.enhance, parser=parser)
        except (ImportError, FileNotFoundError) as e:   # no PyTorch / GFPGAN (e.g. the Windows app)
            print(f"Enhancer not available ({e.__class__.__name__}); running without it.")
    if args.image:
        run_image(args, app, swapper, src, skin, enhancer)
    elif args.keyframes:
        run_keyframes(args, app, swapper, src, skin, parser)
    elif args.aligned:
        run_aligned(args, app, swapper, src, skin, enhancer)
    else:
        run_live(args, app, swapper, src, skin, enhancer)


if __name__ == "__main__":
    main()
