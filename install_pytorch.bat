@echo off

REM Install PyTorch, TorchVision, and Torchaudio with CUDA 11.8 support

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b 1
)

echo Installing PyTorch, TorchVision, and Torchaudio...

pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

if %errorlevel% neq 0 (
    echo Installation failed.
    exit /b %errorlevel%
) else (
    echo Installation completed successfully.
)

pause
