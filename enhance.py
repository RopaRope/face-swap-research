"""Face enhancement after the swap: GFPGAN v1.4 restoration + BiSeNet face-parsing mask.

For each swapped face:
  1. align a 512x512 FFHQ-style crop from the 5 landmarks,
  2. restore it with GFPGAN (fixes inswapper's 128px softness),
  3. parse it with BiSeNet to get a precise mask of skin/brows/eyes/nose/mouth
     (excludes hair, neck, background, clothes),
  4. feather the mask and paste the restored face back.

Runs on PyTorch MPS (Apple GPU) when available, else CPU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

MODELS = Path(__file__).resolve().parent / "models"

from face_parse import FACE_CLASSES, FFHQ_512  # noqa: F401  (re-exported)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class FaceParser:
    """BiSeNet face parsing (CelebAMask-HQ, 19 classes) on an aligned face crop."""

    def __init__(self, parsing_path: Path = MODELS / "parsing_bisenet.pth"):
        from facexlib.parsing.bisenet import BiSeNet

        self.dev = _device()
        self.net = BiSeNet(num_class=19)
        self.net.load_state_dict(torch.load(str(parsing_path), map_location="cpu", weights_only=True))
        self.net.eval().to(self.dev)
        self._mean, self._std = _MEAN.to(self.dev), _STD.to(self.dev)

    @torch.inference_mode()
    def mask(self, crop_bgr: np.ndarray, classes, size: int | None = None) -> np.ndarray:
        """Binary float mask (crop resolution) of the given classes. `size` runs the net smaller."""
        h, w = crop_bgr.shape[:2]
        inp = cv2.resize(crop_bgr, (size, size)) if size and size != w else crop_bgr
        x = torch.from_numpy(inp[..., ::-1].copy()).permute(2, 0, 1)[None].float().to(self.dev) / 255.0
        parse = self.net((x - self._mean) / self._std)[0].argmax(1)[0]
        keep = torch.zeros_like(parse, dtype=torch.bool)
        for c in classes:
            keep |= parse == c
        m = keep.float().cpu().numpy()
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) if m.shape != (h, w) else m


class FaceEnhancer:
    def __init__(self, gfpgan_path: Path = MODELS / "GFPGANv1.4.pth", weight: float = 1.0,
                 parser: FaceParser | None = None):
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        self.dev = _device()
        self.weight = weight   # 1.0 = fully restored face, 0.5 = half blend with the raw swap

        self.gfpgan = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                                    decoder_load_path=None, fix_decoder=False, num_mlp=8,
                                    input_is_latent=True, different_w=True, narrow=1, sft_half=True)
        sd = torch.load(str(gfpgan_path), map_location="cpu", weights_only=False)
        self.gfpgan.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
        self.gfpgan.eval().to(self.dev)
        # fp16 on Apple GPU: 327 -> 275 ms per face on an M1 (CoreML/ONNX paths were no faster).
        self.dtype = torch.float16 if self.dev.type == "mps" else torch.float32
        self.gfpgan.to(self.dtype)

        self.parser = parser or FaceParser()

    @torch.inference_mode()
    def _restore(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(crop_bgr[..., ::-1].copy()).permute(2, 0, 1)[None].float().to(self.dev)
        x = ((x / 255.0 - 0.5) / 0.5).to(self.dtype)
        # randomize_noise=False: fixed StyleGAN noise, otherwise texture flickers frame to frame.
        y = self.gfpgan(x, return_rgb=False, randomize_noise=False)[0].float()
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y[..., ::-1].copy()

    def _face_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        m = self.parser.mask(crop_bgr, FACE_CLASSES)
        m = cv2.erode(m, np.ones((9, 9), np.uint8))
        return cv2.GaussianBlur(m, (31, 31), 0)

    def enhance(self, frame: np.ndarray, faces) -> np.ndarray:
        out = frame
        h, w = frame.shape[:2]
        for f in faces:
            M, _ = cv2.estimateAffinePartial2D(f.kps.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
            if M is None:
                continue
            crop = cv2.warpAffine(out, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            restored = self._restore(crop)
            if self.weight < 1:
                restored = cv2.addWeighted(restored, self.weight, crop, 1 - self.weight, 0)
            mask = self._face_mask(restored)

            # Paste back only within the face's bounding region to keep it cheap.
            IM = cv2.invertAffineTransform(M)
            corners = cv2.transform(np.array([[[0, 0], [511, 0], [0, 511], [511, 511]]], np.float32), IM)[0]
            x1, y1 = np.floor(corners.min(0)).astype(int).clip(0)
            x2, y2 = np.ceil(corners.max(0)).astype(int)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 <= x1 or y2 <= y1:
                continue
            IM_roi = IM.copy()
            IM_roi[:, 2] -= [x1, y1]
            size = (x2 - x1, y2 - y1)
            face_back = cv2.warpAffine(restored, IM_roi, size, flags=cv2.INTER_LINEAR)
            mask_back = cv2.warpAffine(mask, IM_roi, size, flags=cv2.INTER_LINEAR)[..., None]
            if out is frame:
                out = frame.copy()
            roi = out[y1:y2, x1:x2].astype(np.float32)
            out[y1:y2, x1:x2] = (face_back * mask_back + roi * (1 - mask_back)).astype(np.uint8)
        return out
