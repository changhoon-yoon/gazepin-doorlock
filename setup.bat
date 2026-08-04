@echo off
REM GazePIN / HeadPIN demo one-shot setup (Windows, needs Python 3.11 + webcam)
where py >nul 2>nul
if errorlevel 1 (
    echo [!] Python launcher not found. Install Python 3.11 from python.org first.
    exit /b 1
)
py -3.11 -c "print('python 3.11 ok')" 2>nul
if errorlevel 1 (
    echo [!] Python 3.11 not installed. mediapipe 0.10.14 needs 3.9-3.12.
    exit /b 1
)
if not exist .venv py -3.11 -m venv .venv
call .venv\Scripts\pip install -r requirements-demo.txt
if not exist weights mkdir weights
if not exist weights\mobileone_s0_gaze.onnx (
    echo [*] downloading gaze model weights...
    curl -L -o weights\mobileone_s0_gaze.onnx https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
)
echo.
echo [OK] Setup complete. Run one of:
echo   run_gazepin.bat   (eye-gaze 2-choice PIN)
echo   run_headpin.bat   (head-turn 4-choice PIN)
