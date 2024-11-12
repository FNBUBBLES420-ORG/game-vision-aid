@echo off

REM Install torch, torchvision, and torchaudio with +cpu option
echo Installing torch, torchvision, and torchaudio with +cpu option...
pip3 install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu/torch_stable.html

REM Check if the installation was successful
if %errorlevel% neq 0 (
    echo Error: Failed to install torch, torchvision, and torchaudio.
    exit /b %errorlevel%
)

echo Installation completed successfully.
