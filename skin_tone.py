"""Shift the target's visible skin (face, neck, ears, arms, hands) toward the source's skin tone.

Runs on the frame *before* the face swap, so the swapped face and the rest of the body end
up in the same tone. Lighting and shading are kept: only the skin's colour statistics in LAB
space move (mean shift on a/b, mean + contrast on L).

Skin is located with MediaPipe's multiclass selfie segmenter (classes: background, hair,
body-skin, face-skin, clothes, other), so walls, furniture, hair and clothes are never touched.
The segmenter costs ~130 ms on an M1 CPU, so in live mode it runs on a background thread and
the main loop uses the most recent mask (refreshed ~7x/s).
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import paths

DEFAULT_SEGMENTER = paths.model("selfie_multiclass_256x256.tflite")
BODY_SKIN, FACE_SKIN = 2, 3


class SkinSegmenter:
    def __init__(self, model_path: Path = DEFAULT_SEGMENTER):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Skin segmenter not found: {model_path}")
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.seg = vision.ImageSegmenter.create_from_options(opts)

    def skin_prob(self, bgr: np.ndarray, work_w: int = 256) -> np.ndarray:
        """Skin probability (face + body) at a reduced working resolution."""
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (work_w, max(1, int(h * work_w / w))), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        cm = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).confidence_masks
        m = np.clip(cm[BODY_SKIN].numpy_view() + cm[FACE_SKIN].numpy_view(), 0, 1).astype(np.float32)
        return m.reshape(m.shape[0], m.shape[1])   # masks come back (H, W, 1)


class AsyncSkinSegmenter:
    """Runs SkinSegmenter on the newest submitted frame in a background thread."""

    def __init__(self, seg: SkinSegmenter):
        self.seg = seg
        self._frame = None
        self._mask = None
        self._cv = threading.Condition()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame: np.ndarray, anchor):
        """anchor = (cx, cy, scale) of the tracked face in this frame, used for motion compensation."""
        with self._cv:
            self._frame = (frame, anchor)
            self._cv.notify()

    def latest(self):
        """(skin mask at working resolution, anchor of the frame it came from) or None."""
        return self._mask

    def _loop(self):
        while True:
            with self._cv:
                while self._frame is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                (frame, anchor), self._frame = self._frame, None
            try:
                self._mask = (self.seg.skin_prob(frame), anchor)
            except Exception:   # segmenter torn down at interpreter exit
                return

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()


def _anchor(face):
    x1, y1, x2, y2 = face.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0))


def _compensate(mask_small: np.ndarray, old, new, full_w: int) -> np.ndarray:
    """Move/scale a stale mask so the face position it was computed at lines up with the current one.
    The whole person is assumed to move with the face (true for head/torso motion)."""
    r = mask_small.shape[1] / full_w
    k = new[2] / old[2]
    M = np.float32([[k, 0, r * new[0] - k * r * old[0]],
                    [0, k, r * new[1] - k * r * old[1]]])
    return cv2.warpAffine(mask_small, M, (mask_small.shape[1], mask_small.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


# Soft window over the aligned crop: where the same-frame parse takes over from the body mask.
_ZONE = cv2.GaussianBlur(cv2.erode(np.ones((512, 512), np.float32), np.ones((41, 41), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0), (41, 41), 0)

# BiSeNet classes recoloured from the current frame: skin, brows, eyes, ears, nose, mouth, lips, neck.
# (Inner-face classes are included so the mask has no holes; the swap replaces them anyway.)
PARSE_SKIN = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]


def _parsed_face_skin(frame: np.ndarray, face, parser, size: int = 384):
    """Skin/ears/neck mask for one face from *this* frame, plus the soft zone it covers."""
    from face_parse import FFHQ_512
    h, w = frame.shape[:2]
    M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
    if M is None:
        return None, None
    crop = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = parser.mask(crop, PARSE_SKIN, size=size)
    zone = _ZONE
    IM = cv2.invertAffineTransform(M)
    m = cv2.warpAffine(cv2.GaussianBlur(m, (7, 7), 0), IM, (w, h), flags=cv2.INTER_LINEAR)
    zone = cv2.warpAffine(zone, IM, (w, h), flags=cv2.INTER_LINEAR)
    return m, zone


def _landmark_limit(shape, face, scale: float = 0.25) -> np.ndarray:
    """Where recolouring may go at most, from the eyes and mouth (which the detector always
    finds): forehead, cheeks, ears, chin and upper neck. Outside it nothing is recoloured, so
    the parser's stray hair/scalp/shirt/background patches can't show."""
    h, w = shape[:2]
    k = face.kps.astype(np.float32)
    eye = (k[0] + k[1]) / 2
    mouth = (k[3] + k[4]) / 2
    d = float(np.linalg.norm(k[1] - k[0])) + 1e-3        # eye distance
    v = mouth - eye                                         # points down the face
    vl = float(np.linalg.norm(v)) + 1e-3
    u = v / vl
    ang = float(np.degrees(np.arctan2(u[0], u[1])))         # head roll
    top = eye - 1.25 * vl * u                               # forehead, below the hairline
    bot = mouth + 1.05 * vl * u                             # just under the chin (not the collar)
    c = (top + bot) / 2 * scale
    half_h = float(np.linalg.norm(bot - top)) / 2 * scale
    half_w = 1.30 * d * scale                               # out to the ears
    sh, sw = max(int(h * scale), 1), max(int(w * scale), 1)
    m = np.zeros((sh, sw), np.float32)
    cv2.ellipse(m, (int(c[0]), int(c[1])), (max(int(half_w), 1), max(int(half_h), 1)),
                -ang, 0, 360, 1.0, -1)
    # Neck: a column under the chin, narrower than the face so it stays off collar and shoulders,
    # fading out downward so there is no hard line where the colouring stops.
    perp = np.array([u[1], -u[0]], np.float32)
    n_top, n_bot = mouth + 0.6 * vl * u, mouth + 2.1 * vl * u
    nw_top, nw_bot = 0.80 * d, 0.95 * d
    neck = np.zeros_like(m)
    poly = np.array([n_top + perp * nw_top, n_top - perp * nw_top,
                     n_bot - perp * nw_bot, n_bot + perp * nw_bot]) * scale
    cv2.fillConvexPoly(neck, poly.astype(np.int32), 1.0)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    t = ((xs / scale - mouth[0]) * u[0] + (ys / scale - mouth[1]) * u[1]) / vl   # chin-down distance
    neck *= np.clip((2.1 - t) / 0.8, 0, 1)
    m = np.maximum(m, neck)
    kf = max(int(0.25 * d * scale) | 1, 3)
    m = cv2.GaussianBlur(m, (kf, kf), 0)
    below = np.clip((t - 0.9) / 0.4, 0, 1)                  # 0 on the face, 1 below the chin
    return (cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(below, (w, h), interpolation=cv2.INTER_LINEAR))


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _person_mask(shape, face, scale: float = 1.0) -> np.ndarray:
    """Soft region around one person: face plus a wide band below it (neck, arms, hands).
    `scale` builds it at a reduced resolution (shape is then the reduced shape)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = face.bbox * scale
    fw = x2 - x1
    cx = (x1 + x2) / 2
    m = np.zeros((h, w), np.float32)
    rx1, rx2 = int(max(cx - 3.2 * fw, 0)), int(min(cx + 3.2 * fw, w))
    ry1 = int(max(y1 - 0.4 * fw, 0))
    m[ry1:, rx1:rx2] = 1.0
    k = max(int(fw * 0.6) | 1, 3)
    return cv2.GaussianBlur(m, (k, k), 0)


def _beard_box(shape, face) -> np.ndarray:
    """Pixels from the nose down inside the face box: excluded from colour sampling."""
    h, w = shape[:2]
    ex = np.zeros((h, w), bool)
    x1, y1, x2, y2 = face.bbox.astype(int)
    ny = int(face.kps[2][1])
    ex[max(ny, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
    return ex


def _robust_stats(px: np.ndarray):
    if len(px) < 50:
        return None
    med = np.median(px, axis=0)
    mad = np.median(np.abs(px - med), axis=0) * 1.4826 + 1e-3
    keep = np.all(np.abs(px - med) < 2.5 * mad, axis=1)   # drop highlights, deep shadow, stubble
    px = px[keep] if keep.sum() >= 50 else px
    return px.mean(axis=0), px.std(axis=0) + 1e-3


def _skin_stats(bgr: np.ndarray, prob: np.ndarray, face=None, thresh: float = 0.8):
    sel = prob > thresh
    if face is not None:
        sel &= ~_beard_box(bgr.shape, face)
    px = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[sel].astype(np.float32)
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    return _robust_stats(px)


class SkinToneMatcher:
    def __init__(self, src_img: np.ndarray, src_face, strength: float = 1.0, smooth: float = 0.3,
                 live: bool = False, segmenter: SkinSegmenter | None = None, parser=None,
                 stats_every: int = 4):
        self.seg = segmenter or SkinSegmenter()
        h, w = src_img.shape[:2]
        prob = cv2.resize(self.seg.skin_prob(src_img), (w, h))
        stats = _skin_stats(src_img, prob, src_face)
        if stats is None:
            raise ValueError("Could not find skin in the source photo")
        self.src_mean, self.src_std = stats
        self.strength = strength
        self.smooth = smooth          # EMA on target stats so colour doesn't flicker frame to frame
        self._tgt = None
        self.last_mask = None
        self.async_seg = AsyncSkinSegmenter(self.seg) if live else None
        self.parser = parser          # optional FaceParser: same-frame face/ear/neck mask
        self.stats_every = stats_every
        self._i = 0
        self.landmark_limit = False   # True: never recolour outside the eye/mouth-derived boundary
        self._below = None

    def _body_mask(self, frame, face):
        """Body-skin mask at working resolution, aligned to the current frame."""
        anchor = _anchor(face)
        if self.async_seg is None:
            return self.seg.skin_prob(frame)
        self.async_seg.submit(frame, anchor)
        res = self.async_seg.latest()
        if res is None:
            return None
        small, old = res
        return _compensate(small, old, anchor, frame.shape[1])

    def apply(self, frame: np.ndarray, faces, body: np.ndarray | None = None) -> np.ndarray:
        """`body`: skin mask (SkinSegmenter.skin_prob) computed on *this* frame, from the
        frame-aligned pipeline. Without it, live mode uses the latest async mask, motion-compensated."""
        if self.strength <= 0 or not faces:
            return frame
        face = _largest(faces)
        prob = self.mask(frame, face, body)
        if prob is None:
            return frame
        self.update_stats(frame, prob, face)
        return self.recolour(frame, prob)

    def mask(self, frame: np.ndarray, face, body: np.ndarray | None = None):
        """Full-resolution soft skin mask for the person around `face`, or None."""
        h, w = frame.shape[:2]
        small = body if body is not None else self._body_mask(frame, face)
        if small is None:
            return None
        small = small * _person_mask(small.shape, face, scale=small.shape[1] / w)
        prob = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.parser is not None:
            fm, zone = _parsed_face_skin(frame, face, self.parser)
            if fm is not None:
                # Union with the (motion-compensated) MediaPipe mask: BiSeNet can label a shaved or
                # bald scalp as background/hat, MediaPipe handles it; either one calling it skin wins.
                prob = np.maximum(fm, prob) * zone + prob * (1 - zone)
        self._below = None
        if self.landmark_limit:
            lim, self._below = _landmark_limit(frame.shape, face)
            prob = prob * lim
        self.last_mask = prob
        return prob

    def update_stats(self, frame: np.ndarray, prob: np.ndarray, face, force: bool = False):
        """Refresh the target's skin colour statistics (EMA, every `stats_every` calls)."""
        self._i += 1
        st = None
        if force or self._tgt is None or self._i % self.stats_every == 0:
            st = _skin_stats(frame, prob, face)
        if st is not None:
            if self._tgt is None:
                self._tgt = st
            else:
                a = self.smooth
                self._tgt = tuple(a * n + (1 - a) * o for n, o in zip(st, self._tgt))

    def recolour(self, frame: np.ndarray, prob: np.ndarray) -> np.ndarray:
        """Move skin under `prob` from the target's tone to the source's, using current stats."""
        if self._tgt is None:
            return frame
        t_mean, t_std = self._tgt

        ys, xs = np.nonzero(prob > 0.05)
        if len(ys) == 0:
            return frame
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = frame[y1:y2, x1:x2]
        # Firm up the mask: half-covered pixels (hair edge, shadowed hands, scalp shine) got only
        # half the lift and showed as orange/brown patches when the tones are far apart.
        p = np.clip((prob[y1:y2, x1:x2] - 0.15) / 0.45, 0, 1)
        m = cv2.GaussianBlur(p * p * (3 - 2 * p), (15, 15), 0)[..., None] * self.strength   # no blocky edges

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        if self._below is not None:
            # Below the chin, only lift pixels that look like the measured skin: the neck column
            # overlaps collars and shirt openings, which the parser can call "neck". (Not on the
            # face itself: its highlights and shadows don't match the average tone.)
            tol = np.maximum(np.array([3.5, 3.0, 3.0], np.float32) * t_std,
                             np.array([18.0, 7.0, 7.0], np.float32))
            z = (cv2.GaussianBlur(lab, (5, 5), 0) - t_mean) / tol
            like = np.exp(-0.5 * (z * z).sum(axis=2))
            gate = cv2.GaussianBlur(np.clip(like * 1.6, 0, 1), (9, 9), 0)
            b = self._below[y1:y2, x1:x2]
            m = m * (1 - b + b * gate)[..., None]
        # Dark, noisy webcam skin: lifting it far amplifies sensor noise into blotches. Smooth
        # lightness edge-preservingly and chroma more strongly (skin colour varies slowly).
        L = cv2.bilateralFilter(lab[..., 0], 7, 12, 5)
        ab = cv2.GaussianBlur(lab[..., 1:], (9, 9), 0)
        ratio_l = float(np.clip(self.src_std[0] / t_std[0], 0.7, 1.3))
        new = np.empty_like(lab)
        nl = (L - t_mean[0]) * ratio_l + self.src_mean[0]
        knee = min(self.src_mean[0] + 1.5 * self.src_std[0], 235.0)
        over = np.maximum(nl - knee, 0)       # roll highlights off smoothly instead of clipping
        nl = np.where(nl > knee, knee + (255 - knee) * (1 - np.exp(-over / (255 - knee))), nl)
        new[..., 0] = nl
        # Fade the colour shift out in near-white highlights so shine doesn't turn orange.
        fade = np.clip((250 - nl) / 40, 0, 1)[..., None]
        new[..., 1:] = ab + (self.src_mean[1:] - t_mean[1:]) * fade
        lab += (new - lab) * m
        out = frame.copy()
        out[y1:y2, x1:x2] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out
