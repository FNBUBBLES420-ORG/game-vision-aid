@echo off

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b 1
)

echo Installing PyTorch and related packages...

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if %errorlevel% neq 0 (
    echo Installation failed.
    exit /b %errorlevel%
) else (
    echo Installation completed successfully.
)

pause
