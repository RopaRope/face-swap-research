#!/usr/bin/env bash
# One-time setup + start on a Runpod pod (runpod/pytorch image, CUDA 12.x).
# Usage (from the folder this repo was copied to, e.g. /workspace/face-swap):
#   bash pod/setup.sh /workspace/face.jpg
# Prints the two links to put into OBS. Re-running is safe (skips finished steps).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?usage: bash pod/setup.sh <source face photo>}"
DATA=/workspace/faceswap-data
export FACESWAP_DATA="$DATA"
MTX_VER=v1.21.4

mkdir -p "$DATA/models" "$DATA/models/buffalo_l" /workspace/bin

echo "== system packages"
if ! command -v ffmpeg >/dev/null || ! ldconfig -p | grep -q libGL.so.1; then
  apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 >/dev/null
fi

echo "== python packages"
# Ubuntu 24.04 blocks system-wide pip (PEP 668): use a venv that still sees the image's PyTorch.
VENV=/workspace/venv
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV"
export PATH="$VENV/bin:$PATH"
if [ ! -f /workspace/.faceswap-pip-done ]; then
  pip install -q --upgrade pip
  pip install -q onnx "onnxruntime-gpu[cuda,cudnn]" opencv-python mediapipe==0.10.31 av \
      scikit-image scipy tqdm prettytable easydict requests cython facexlib \
      addict yapf lmdb future
  pip install -q --no-deps --no-build-isolation insightface==2.1
  pip install -q --no-deps basicsr==1.4.2 gfpgan==1.3.8
  # basicsr 1.4.2 imports names newer torchvision / Python removed
  SP=$(python -c "import site;print(site.getsitepackages()[0])")
  sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/' "$SP/basicsr/data/degradations.py"
  sed -i 's/^from distutils.version import LooseVersion$/from packaging.version import Version as LooseVersion/' "$SP/basicsr/archs/arch_util.py"
  touch /workspace/.faceswap-pip-done
fi

echo "== models (official sources, checked by SHA-256)"
python - <<'PY'
import faceswap_app as a, paths, zipfile, os, shutil
for name, url, size, digest in a.MODELS:
    if not (paths.MODELS / name).exists():
        print("downloading", name); a.download(url, paths.MODELS / name, size, digest)
buf = paths.INSIGHTFACE_ROOT / "models" / "buffalo_l"
if not all((buf / f).exists() for f in a.BUFFALO_FILES):
    name, url, size, digest = a.BUFFALO
    z = paths.INSIGHTFACE_ROOT / "models" / name
    a.download(url, z, size, digest)
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if m.endswith(".onnx"):
                with zf.open(m) as s, open(buf / os.path.basename(m), "wb") as d:
                    shutil.copyfileobj(s, d)
    z.unlink()
PY
# BiSeNet face parser (CodeFormer release), run with PyTorch on the GPU
PARSE="$DATA/models/parsing_bisenet.pth"
if [ ! -f "$PARSE" ]; then
  wget -q -O "$PARSE.part" https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth
  echo "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567  $PARSE.part" | sha256sum -c - && mv "$PARSE.part" "$PARSE"
fi
mkdir -p models && ln -sf "$PARSE" models/parsing_bisenet.pth
# GFPGAN v1.4 face restoration (official TencentARC release)
GFP="$DATA/models/GFPGANv1.4.pth"
if [ ! -f "$GFP" ]; then
  wget -q -O "$GFP.part" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
  echo "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad  $GFP.part" | sha256sum -c - && mv "$GFP.part" "$GFP"
fi
ln -sf "$GFP" models/GFPGANv1.4.pth

echo "== MediaMTX media server"
if [ ! -x /workspace/bin/mediamtx ]; then
  wget -q -O /tmp/mtx.tgz "https://github.com/bluenviron/mediamtx/releases/download/${MTX_VER}/mediamtx_${MTX_VER}_linux_amd64.tar.gz"
  tar -xzf /tmp/mtx.tgz -C /workspace/bin mediamtx
fi
CRED=/workspace/faceswap-data/stream-credentials
if [ ! -f "$CRED" ]; then
  echo "PUB_PASS=$(head -c 12 /dev/urandom | base64 | tr -dc A-Za-z0-9)" > "$CRED"
  echo "READ_PASS=$(head -c 12 /dev/urandom | base64 | tr -dc A-Za-z0-9)" >> "$CRED"
fi
. "$CRED"
cat > /workspace/bin/mediamtx.yml <<YML
logLevel: warn
rtsp: yes
rtspTransports: [tcp]
rtspAddress: :8554
rtmp: yes
rtmpAddress: :1935
hls: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
authMethod: internal
authInternalUsers:
  - user: any
    ips: ['127.0.0.1', '::1']
    permissions: [{action: publish}, {action: read}]
  - user: obs
    pass: ${PUB_PASS}
    permissions: [{action: publish, path: live}]
  - user: viewer
    pass: ${READ_PASS}
    permissions: [{action: read, path: swapped}]
paths:
  live: {}
  swapped: {}
YML

echo "== starting"
pkill -f "bin/mediamtx" 2>/dev/null || true
pkill -f "pod/pod_server.py" 2>/dev/null || true
sleep 1
nohup /workspace/bin/mediamtx /workspace/bin/mediamtx.yml > /workspace/mediamtx.log 2>&1 &
cp "$SRC" "$DATA/source.jpg" 2>/dev/null || true
nohup python pod/pod_server.py --source "$DATA/source.jpg" > /workspace/pod_server.log 2>&1 &
sleep 2

IP="${RUNPOD_PUBLIC_IP:-<pod-public-ip>}"
RTMP_PORT="${RUNPOD_TCP_PORT_1935:-<public port for 1935>}"
RTSP_PORT="${RUNPOD_TCP_PORT_8554:-<public port for 8554>}"
cat <<EOF

=====================================================================
 OBS  > Settings > Stream > Service: Custom
   Server:      rtmp://${IP}:${RTMP_PORT}
   Stream Key:  live?user=obs&pass=${PUB_PASS}
 OBS  > Media Source (uncheck "Local File"):
   Input:       rtsp://viewer:${READ_PASS}@${IP}:${RTSP_PORT}/swapped
   Input Format: rtsp     ffmpeg options: rtsp_transport=tcp
 Logs: tail -f /workspace/pod_server.log
=====================================================================
EOF
