@echo off
echo MAKE SURE TO HAVE THE WHL DOWNLOADED BEFORE YOU CONTINUE!!!
pause
echo Click the link to download the WHL:
echo https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo Installing CuPy from WHL...
pip install https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo All packages installed successfully!
pause
