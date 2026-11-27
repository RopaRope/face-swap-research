"""GPU face-swap relay for the Runpod server.

OBS on your computer streams webcam + microphone to MediaMTX on the pod (RTMP, path `live`).
This service reads that stream, swaps the face on every frame on the GPU, burns in the
"SYNTHETIC / NOT the real person" label, and publishes video + the original audio, kept in sync,
as path `swapped`. OBS plays `swapped` in a Media Source and sends it to calls via its virtual
camera (and the audio via a virtual audio device).

Frames are processed by a small pool of workers and written back in order. If the pod falls
behind, video frames are dropped (never audio), so the delay stays bounded.
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # pods expose ~96 cores but allow ~8: more threads only thrash

import av
import cv2

cv2.setNumThreads(4)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import swap_live as sl                                   # noqa: E402
from skin_tone import SkinSegmenter, SkinToneMatcher     # noqa: E402


class Swapper:
    """The frame-aligned pipeline (same as `swap_live.py --aligned`), safe to call from threads."""

    def __init__(self, source: str, det_size: int, skin_strength: float, enhance: float,
                 body_skin: bool = False):
        print("providers:", sl.providers(), flush=True)
        self.app, self.swapper = sl.load_models(sl.DEFAULT_SWAPPER, det_size, fp16=False)
        self.src = sl.source_face(self.app, source)
        self.parser = sl.load_parser() if skin_strength > 0 else None
        self.skin = (SkinToneMatcher(cv2.imread(source), self.src, strength=skin_strength, live=False,
                                     parser=self.parser) if skin_strength > 0 else None)
        self.enhancer = None
        if enhance > 0:     # GFPGAN restores the 128 px swap to a sharp 512 px face, every frame
            from enhance import FaceEnhancer
            self.enhancer = FaceEnhancer(weight=enhance, parser=self.parser)
        self.body_skin = body_skin   # MediaPipe body mask: ~120 ms of CPU per frame, only for neck/arms
        self.local = threading.local()
        self.lock = threading.Lock()

    def _segmenter(self) -> SkinSegmenter:
        if not hasattr(self.local, "seg"):
            self.local.seg = SkinSegmenter()
        return self.local.seg

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        faces = sl.detect(self.app, bgr)
        body = None
        if self.skin is not None and faces:
            if self.body_skin:
                body = self._segmenter().skin_prob(bgr)
            else:   # face/ears/neck still come from the GPU face parser
                h, w = bgr.shape[:2]
                body = np.zeros((max(1, h * 256 // w), 256), np.float32)
        out = bgr
        if faces:
            if self.skin is not None:
                with self.lock:             # only the skin-tone statistics are shared state
                    out = self.skin.apply(bgr, faces, body=body)
            # swap + GFPGAN run in parallel across workers (ONNX Runtime / PyTorch are thread-safe)
            out, _ = sl.swap_frame(out, self.app, self.swapper, self.src, faces, enhancer=self.enhancer)
        return sl.watermark(out)            # always: every frame says it is synthetic


def open_input(url: str):
    while True:
        try:
            c = av.open(url, options={"rtsp_transport": "tcp", "fflags": "nobuffer",
                                      "flags": "low_delay", "max_delay": "0"}, timeout=(10, 10))
            # A stream that is still starting up can report 0x0; the encoder then hangs.
            if c.streams.video and c.streams.video[0].codec_context.width > 0:
                return c
            c.close()
        except av.error.FFmpegError as e:
            print(f"waiting for the webcam stream ({e.__class__.__name__})...", flush=True)
        time.sleep(2)


def run_once(args, swap: Swapper):
    inp = open_input(args.input)
    vin = inp.streams.video[0]
    ain = inp.streams.audio[0] if inp.streams.audio else None
    vin.thread_type = "AUTO"
    print(f"input: {vin.codec_context.width}x{vin.codec_context.height} video"
          f"{', ' + str(ain.codec_context.sample_rate) + ' Hz audio' if ain else ', no audio'}", flush=True)

    out = av.open(args.output, "w", format="rtsp", options={"rtsp_transport": "tcp"})
    vout = out.add_stream("libx264", rate=args.fps)
    vout.width, vout.height = vin.codec_context.width, vin.codec_context.height
    vout.pix_fmt = "yuv420p"
    vout.bit_rate = args.bitrate
    # Keyframe every 4 s: at 1/s each one was a visible once-a-second pulse at this bitrate.
    vout.options = {"preset": "veryfast", "tune": "zerolatency", "g": str(args.fps * 4)}
    vout.time_base = vin.time_base
    vout.codec_context.time_base = vin.time_base   # input clock: paced frames never share a tick
    aout = None
    if ain:
        aout = out.add_stream("aac", rate=ain.codec_context.sample_rate)
        aout.layout = ain.codec_context.layout
        aout.time_base = ain.time_base

    pool = ThreadPoolExecutor(args.workers)
    pending: collections.deque = collections.deque()     # (video pts, future)
    audio_q: collections.deque = collections.deque()     # decoded audio frames waiting for video
    last_vpts_s = -1.0
    last_pts = None
    next_t = -1.0          # pacing: take frames on an even time grid so motion doesn't stutter
    shown = dropped = 0
    t_stat = time.time()

    def write_audio_up_to(t_s: float):
        while audio_q and (audio_q[0].pts * float(ain.time_base)) <= t_s:
            for p in aout.encode(audio_q.popleft()):
                out.mux(p)

    def drain(block: bool):
        nonlocal last_vpts_s, shown, last_pts
        while pending and (block or pending[0][1].done()):
            pts, fut = pending.popleft()
            img = fut.result()
            if last_pts is not None and pts <= last_pts:   # encoder needs strictly rising timestamps
                continue
            last_pts = pts
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = pts, vin.time_base
            t_s = pts * float(vin.time_base)
            if aout is not None:
                write_audio_up_to(t_s)      # interleave: audio up to this video time, then the frame
            for p in vout.encode(frame):
                out.mux(p)
            last_vpts_s = t_s
            shown += 1
            block = False

    try:
        for pkt in inp.demux(*(s for s in (vin, ain) if s is not None)):
            if pkt.stream is ain:
                for af in pkt.decode():
                    af.pts = af.pts if af.pts is not None else 0
                    audio_q.append(af)
                # never let audio get far ahead if video stalls (e.g. no frames decoded yet)
                if audio_q and last_vpts_s >= 0:
                    write_audio_up_to(last_vpts_s + args.max_lag)
                continue
            for vf in pkt.decode():
                if vf.pts is None:
                    continue
                t = vf.pts * float(vin.time_base)
                if args.fps and t < next_t:
                    continue
                next_t = max(next_t + 1.0 / args.fps, t - 0.5 / args.fps) if args.fps else 0
                if len(pending) >= args.workers * 2:     # behind: drop this video frame, keep audio
                    dropped += 1
                    continue
                pending.append((vf.pts, pool.submit(swap, vf.to_ndarray(format="bgr24"))))
            drain(block=False)      # never wait here: a full queue drops frames, so delay stays bounded
            if time.time() - t_stat > 5:
                print(f"{shown / (time.time() - t_stat):.1f} fps out, {dropped} dropped, "
                      f"{len(pending)} in flight", flush=True)
                shown, t_stat = 0, time.time()
        drain(block=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for s in (vout, aout):
            if s is not None:
                try:
                    for p in s.encode(None):
                        out.mux(p)
                except Exception:
                    pass
        out.close()
        inp.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="photo of the consenting person whose face is swapped in")
    ap.add_argument("--input", default="rtsp://127.0.0.1:8554/live")
    ap.add_argument("--output", default="rtsp://127.0.0.1:8554/swapped")
    ap.add_argument("--fps", type=int, default=12,
                    help="frames per second taken from the webcam stream, evenly spaced")
    ap.add_argument("--bitrate", type=int, default=1_800_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--skin", type=float, default=1.0)
    ap.add_argument("--body-skin", action="store_true",
                    help="also recolour neck/arms/hands with MediaPipe (costs ~120 ms CPU per frame)")
    ap.add_argument("--enhance", type=float, default=0.5,
                    help="GFPGAN face restoration strength (0 = off, 1 = full)")
    ap.add_argument("--max-lag", type=float, default=1.0,
                    help="seconds audio may run ahead of video before it is sent anyway")
    args = ap.parse_args()
    swap = Swapper(args.source, args.det_size, args.skin, args.enhance, args.body_skin)
    swap(np.zeros((720, 1280, 3), np.uint8))       # warm up GPU kernels before the stream starts
    print("ready: waiting for OBS to stream to path 'live'", flush=True)
    while True:
        try:
            run_once(args, swap)
            print("stream ended; waiting for the next one", flush=True)
        except Exception as e:                       # keep serving across OBS restarts
            print(f"stream error: {e!r}; restarting", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
