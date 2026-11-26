"""BiSeNet face parsing (CelebAMask-HQ, 19 classes) on ONNX Runtime: no PyTorch needed.

The ONNX file is made once from CodeFormer's `parsing_bisenet.pth` with `export_onnx()` (that step
needs torch + facexlib; the Windows app ships or downloads the finished .onnx instead).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# facexlib's 5-point template for 512x512 FFHQ-aligned faces (what GFPGAN/BiSeNet were trained on).
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)

# Classes kept in a face paste mask:
# 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears, 10 nose, 11 mouth, 12/13 lips.
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
EXPORT_SIZE = 384   # fixed input size of the exported model (what the live pipeline uses)


class OnnxFaceParser:
    """Same interface as enhance.FaceParser: `mask(crop_bgr, classes, size)`."""

    def __init__(self, path: Path, providers: list | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so,
                                         providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = shape[2] if isinstance(shape[2], int) else EXPORT_SIZE

    def mask(self, crop_bgr: np.ndarray, classes, size: int | None = None) -> np.ndarray:
        """Binary float mask (crop resolution) of the given classes. `size` is accepted for
        compatibility; the model runs at its exported size."""
        h, w = crop_bgr.shape[:2]
        inp = cv2.resize(crop_bgr, (self.size, self.size)) if (h, w) != (self.size, self.size) else crop_bgr
        x = (inp[..., ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        parse = self.sess.run(None, {self.inp: x})[0][0]
        m = np.isin(parse, classes).astype(np.float32)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) if m.shape != (h, w) else m


def export_onnx(pth: Path, out: Path, size: int = EXPORT_SIZE):
    """Convert parsing_bisenet.pth to an ONNX model that outputs the per-pixel class (argmax)."""
    import torch
    from facexlib.parsing.bisenet import BiSeNet

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = BiSeNet(num_class=19)
            self.net.load_state_dict(torch.load(str(pth), map_location="cpu", weights_only=True))

        def forward(self, x):
            return self.net(x)[0].argmax(1).to(torch.uint8)

    m = Wrapped().eval()
    torch.onnx.export(m, torch.zeros(1, 3, size, size), str(out), input_names=["input"],
                      output_names=["parse"], opset_version=17, dynamo=False)


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent / "models"
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "parsing_bisenet.pth"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "parsing_bisenet.onnx"
    export_onnx(src, dst)
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
