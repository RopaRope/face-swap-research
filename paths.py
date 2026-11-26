"""Where models and settings live, for both the source checkout and the packaged Windows app.

Source checkout: models/ next to the code; insightface's buffalo_l in ~/.insightface/models/.
Packaged app (PyInstaller): everything downloaded goes to %LOCALAPPDATA%\\FaceSwapResearch (or
~/.faceswap-research elsewhere); small files shipped inside the app are read from the bundle.
Set FACESWAP_DATA to use another folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))     # files shipped inside the packaged app


def _data_dir() -> Path:
    if os.environ.get("FACESWAP_DATA"):
        return Path(os.environ["FACESWAP_DATA"]).expanduser()
    if not FROZEN:
        return ROOT
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "FaceSwapResearch" if base else Path.home() / ".faceswap-research"


DATA = _data_dir()
MODELS = DATA / "models"
# insightface looks for <root>/models/buffalo_l
INSIGHTFACE_ROOT = DATA if (FROZEN or os.environ.get("FACESWAP_DATA")) else Path.home() / ".insightface"


def model(name: str) -> Path:
    """A model file: the downloaded copy if present, else one shipped in the bundle."""
    p = MODELS / name
    if p.exists():
        return p
    b = BUNDLE / "models" / name
    return b if b.exists() else p
