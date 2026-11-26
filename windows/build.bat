@echo off
rem Builds dist\FaceSwapResearch\FaceSwapResearch.exe (+ a zip of that folder).
rem Needs: Python 3.12 (64-bit, python.org) and "Visual Studio Build Tools" with the
rem "Desktop development with C++" workload (insightface compiles a small C++ part on install).
rem Run from the project folder:   windows\build.bat
setlocal
cd /d "%~dp0\.."

if not exist models\parsing_bisenet.onnx (
  echo models\parsing_bisenet.onnx is missing. Copy it from the Mac project folder ^(it ships inside the app^).
  exit /b 1
)

where py >nul 2>nul || (echo Python launcher "py" not found. Install Python 3.12 from python.org. & exit /b 1)
if not exist .venv-win (
  py -3.12 -m venv .venv-win || exit /b 1
)
call .venv-win\Scripts\activate.bat
python -m pip install --upgrade pip wheel setuptools || exit /b 1
python -m pip install -r windows\requirements-windows.txt || exit /b 1
python -m pip install --no-deps --no-build-isolation insightface==2.1 || (
  echo.
  echo insightface failed to build. Install "Visual Studio Build Tools" ^(free, from
  echo visualstudio.microsoft.com/visual-cpp-build-tools^) with "Desktop development with C++",
  echo then run this script again.
  exit /b 1
)

python -m PyInstaller --noconfirm --clean windows\FaceSwapResearch.spec || exit /b 1
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\FaceSwapResearch -DestinationPath dist\FaceSwapResearch-windows.zip"
echo.
echo Done: dist\FaceSwapResearch\FaceSwapResearch.exe  (and dist\FaceSwapResearch-windows.zip)
