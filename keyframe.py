"""Keyframe face swap: the full swap on some frames, cheap landmark-driven warps for the rest.

The full pipeline (skin mask, recolour, inswapper) runs on "keyframes" in a worker thread, as
often as the machine manages. Every camera frame is still shown: the display runs a fixed
`delay` behind the camera, re-detects the face landmarks on each frame, warps the swapped face
from the keyframe nearest in time onto them (that keyframe may be slightly *later* than the frame,
which the delay makes possible), and recolours skin with that keyframe's mask moved to follow the
face. Head motion is smooth at camera rate; mouth/eye changes update at the keyframe rate.
"""

from __future__ import annotations

import collections
import queue
import threading
import time

import cv2
import numpy as np

from face_parse import FACE_CLASSES, FFHQ_512
from skin_tone import _anchor, _compensate, _largest

PATCH = 512


class FrameBus:
    """Reads the camera in its own thread and keeps the last few seconds of frames."""

    def __init__(self, cap, keep: int = 120):
        self.cap = cap
        self.buf: collections.deque = collections.deque(maxlen=keep)   # (seq, time, frame)
        self.cv = threading.Condition()
        self.seq = 0
        self._handed = -1
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.cv:
                self.buf.append((self.seq, time.perf_counter(), frame))
                self.seq += 1
                self.cv.notify_all()

    def read_stamped(self, timeout: float = 1.0):
        """Newest frame not handed out yet (feeds the keyframe worker): (time, frame) or None."""
        with self.cv:
            if not self.cv.wait_for(lambda: self.buf and self.buf[-1][0] > self._handed, timeout):
                return None
            s, t, f = self.buf[-1]
            self._handed = s
            return t, f

    def at(self, seq: int, timeout: float = 1.0):
        """Frame number `seq` (waits for it; the oldest kept one if it already fell out)."""
        with self.cv:
            if not self.cv.wait_for(lambda: self.buf and self.buf[-1][0] >= seq, timeout):
                return None
            idx = max(seq - self.buf[0][0], 0)
            return self.buf[idx] if idx < len(self.buf) else None

    def seq_before(self, t_max: float) -> int:
        """Newest frame number captured at or before `t_max` (-1 if none)."""
        with self.cv:
            for s, t, _ in reversed(self.buf):
                if t <= t_max:
                    return s
            return -1

    def close(self):
        self.stop.set()


class Keyframe:
    __slots__ = ("t", "out", "patch", "pmask", "prob", "anchor")

    def __init__(self, t, out, patch=None, pmask=None, prob=None, anchor=None):
        self.t, self.out, self.patch, self.pmask, self.prob, self.anchor = t, out, patch, pmask, prob, anchor


def _align(kps) -> np.ndarray | None:
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), FFHQ_512, method=cv2.LMEDS)
    return M


class KeyframeWorker:
    """Runs the full swap on the newest frame (with its own skin mask) over and over."""

    def __init__(self, frames, detect, swap_face, skin, parser, max_rate: float = 0.0):
        self.frames = frames          # AlignedFrames fed from a FrameBus
        self.detect = detect          # frame -> faces
        self.swap_face = swap_face    # (frame, faces) -> swapped frame (new array)
        self.skin = skin
        self.parser = parser
        self.min_dt = 1.0 / max_rate if max_rate > 0 else 0.0
        self.keys: collections.deque = collections.deque(maxlen=16)
        self.lock = threading.Lock()
        self.count = 0
        self.busy_s = 0.0
        self.enabled = True
        self.skin_on = True
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last = 0.0
        while not self.stop.is_set():
            wait = last + self.min_dt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            try:
                t, frame, body = self.frames.get(timeout=1.0)
            except queue.Empty:
                continue
            last = t0 = time.perf_counter()
            try:
                key = self._make(t, frame, body)
            except Exception as e:          # keep the display alive; report and carry on
                print(f"keyframe failed: {e!r}", flush=True)
                continue
            with self.lock:
                self.keys.append(key)
            self.count += 1
            self.busy_s += time.perf_counter() - t0

    def _make(self, t, frame, body) -> Keyframe:
        faces = self.detect(frame) if self.enabled else []
        if not faces:
            return Keyframe(t, frame)
        face = _largest(faces)
        pre, prob = frame, None
        if self.skin is not None and self.skin_on:
            prob = self.skin.mask(frame, face, body)
            if prob is not None:
                self.skin.update_stats(frame, prob, face)
                pre = self.skin.recolour(frame, prob)
        out = self.swap_face(pre, [face])
        M = _align(face.kps)
        if M is None:
            return Keyframe(t, out)
        patch = cv2.warpAffine(out, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pm = self.parser.mask(patch, FACE_CLASSES, size=384)
        pm = cv2.GaussianBlur(cv2.erode(pm, np.ones((9, 9), np.uint8)), (31, 31), 0)
        return Keyframe(t, out, patch, pm, prob, _anchor(face))

    def nearest(self, t: float) -> Keyframe | None:
        with self.lock:
            if not self.keys:
                return None
            return min(self.keys, key=lambda k: abs(k.t - t))

    def close(self):
        self.stop.set()


def render(frame: np.ndarray, t: float, faces, key: Keyframe | None, skin) -> np.ndarray:
    """Output for one camera frame, built from the nearest keyframe."""
    if key is None:
        return frame.copy()
    if key.t == t:
        return key.out.copy()          # this frame *is* the keyframe
    if key.patch is None or not faces:
        return frame.copy()
    face = _largest(faces)
    h, w = frame.shape[:2]
    out = frame
    if skin is not None and key.prob is not None:
        # The keyframe's skin mask, moved with the face (it is at most ~1 keyframe interval old).
        out = skin.recolour(frame, _compensate(key.prob, key.anchor, _anchor(face), w))
    if out is frame:
        out = frame.copy()

    M = _align(face.kps)
    if M is None:
        return out
    IM = cv2.invertAffineTransform(M)
    corners = cv2.transform(np.array([[[0, 0], [PATCH - 1, 0], [0, PATCH - 1], [PATCH - 1, PATCH - 1]]],
                                     np.float32), IM)[0]
    x1, y1 = np.floor(corners.min(0)).astype(int).clip(0)
    x2, y2 = np.ceil(corners.max(0)).astype(int)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return out
    IM_roi = IM.copy()
    IM_roi[:, 2] -= [x1, y1]
    size = (x2 - x1, y2 - y1)
    face_back = cv2.warpAffine(key.patch, IM_roi, size, flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(key.pmask, IM_roi, size, flags=cv2.INTER_LINEAR)[..., None]
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (face_back * m + roi * (1 - m)).astype(np.uint8)
    return out
