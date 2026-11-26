# PyInstaller spec for the Windows app. Build with windows\build.bat.
# One folder (not one file): mediapipe and onnxruntime are large, and a one-file exe would unpack
# them to a temp folder on every start.
import os
from PyInstaller.utils.hooks import collect_all

TOP = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(TOP, "models", "parsing_bisenet.onnx"), "models")]
binaries, hiddenimports = [], ["face_parse", "skin_tone", "keyframe", "paths", "swap_live"]
for pkg in ("mediapipe", "onnxruntime", "insightface", "pyvirtualcam", "pygrabber", "comtypes",
            "skimage", "scipy"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

a = Analysis([os.path.join(TOP, "faceswap_app.py")], pathex=[TOP], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports,
             excludes=["torch", "torchvision", "gfpgan", "basicsr", "facexlib", "tkinter",
                       "matplotlib", "enhance"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FaceSwapResearch", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="FaceSwapResearch")
