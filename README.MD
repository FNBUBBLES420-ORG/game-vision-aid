<p align="center">
  <img src="https://github.com/FNBUBBLES420-ORG/game-vision-aid/blob/main/banner/Game_Vision_Aid.png" alt="Game Vision Aid Banner" width="400"/>
</p>


<div align="center">

# **🚨** ***READ EVERYTHING CAREFULLY !!!*** **🚨**

# **Including**: `Readme.md`, `License.md`, `Code_of_Conduct.md`, `Security.md`.

</div>


# 🎮 GameVisionAid

**GameVisionAid** is an accessibility tool designed to assist visually impaired gamers by enhancing visual cues in video games. It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.

## 📥 How to Download the Repo (First-Time Users)

Click the link to read [**Instructions**](https://www.gitprojects.fnbubbles420.org/how-to-download-repos) 📄.

## Discord Link
- head to `game-vision-aid` channel if you need help
- make sure to check out the rest of the server
# **[Discord Link](https://discord.gg/WvqTu92aMs)**

## Support the Project ⭐

If you find this project useful, please give it a star! Your support is appreciated and helps keep the project growing. 🌟

## Contact 

**[Contact](https://github.com/FNBUBBLES420-ORG/game-vision-aid/blob/main/Contact.md)**


## 📂 Project Structure

```
game-vision-aid/ 
├── .github/                              # For Issues
├── AMD-GPU-Only folder                   # AMD GPU folder
├──    └── IMPORTANT_FOR_AMD_GPU.md       # Important Readme.md for AMD GPU USERS
├──    └── amd_gpu_requirements.bat       # AMD GPU requirements batchfile
├──    └── amdgpu-launcher.bat            # AMD GPU launcher batchfile
├──    └── config-launcher.bat            # AMD config-launcher batchfile
├──     └── config.py                     # AMD Python Configuration file
├──     └── main.py                       # AMD Main script file
├──     └── readme.md                     # Main Readme.md for AMD GPU USERS
├──     └── requirements.txt              # AMD List of required Python packages
├── banner folder                         # Banner Image PNG
├── models/                               # Where the models go
├──   └── fn_v5.pt                        # Custom YOLOv5 model (if available)
├──   └── model_v5s.pt                    # Custom YOLOv5 model (if available)
├──   └── fn_v5v5480480Half.onnx          # Custom ONNX model (if available) nvidia gpu 2060 super
├──   └── model_fn_v5v5320320Half.onnx    # Custom ONNX model (if available) nvidia gpu 2060 super
├──   └── model_v5sv5320320Half.onnx      # Custom ONNX model (if available) nvidia gpu 2060 super
├──   └── model_v5sv5480480Half.onnx      # Custom ONNX model (if available) nvidia gpu 2060 super
├──  Export-Models                        # Export Models Folder
├──    └── models                         # part of the exporting process
├──    └── ultralytics1/utils             # ultralytics/utils
├──    └── utils                          # utils folder
├──    └── export.py                      # export script
├──    └── update_ultralytics.bat         # update ultralytics batchfile
├── src/                                  # Part of RUST Application
├──   └── main.rs                         # Part of RUST Application
├── .gitignore                            # gitignore
├── CODE_OF_CONDUCT.md                    # CODE_OF_CONDUCT.md
├── Cargo.lock                            # Part of RUST Application
├── Cargo.toml                            # Part of RUST Application
├── Contact.md                            # Contact
├── LICENSE.md                            # LICENSE
├── List-of-Colors.md                     # List of colors for the Bounding Boxes
├── README.md                             # Project documentation
├── SECURITY.md                           # Security documentation
├── config.py                             # Configuration file 
├── cudnn_instructions.js                 # Instructions in JavaScript
├── cupy_cuda11x.bat                      # Cupy Cuda 11x Batchfile
├── install_python.bat                    # Python 3.11.6 Batchfile
├── install_pytorch.bat                   # Pytorch Build wheel for the script GPU
├── install_pytorch_cpu_only.bat          # Pytorch Build for CPU 
├── main.py                               # Main script file
├── model_v5s.pt                          # pt file goes in modelS folder
├── model_v5sv5480480Half.onnx            # onnx file goes in modelS folder
├── model_fn_v5v5320320Half.onnx          # onnx file goes in modelS folder
├── models-fn_v5.zip                      # Compressed models to in main models folder 
├── nodejs-instructions.ps1               # Instructions in Powershell
├── notes.txt                             # Read Notes.txt
├── requirements.bat                      # Installs list of required Python packages
├── requirements.txt                      # List of required Python packages
├── update_ultralytics.bat                # Update Batch Script for Ultralytics
```


## 🚀 Features

- 🖥️ **Real-Time Screen Capture**: Captures your screen in real-time using `BetterCam` for fast and efficient screen capturing.
- 🎯 **Object Detection**: Utilizes `YOLOv5` for detecting enemies in video games.
- 🟩 **Customizable Overlays**: Allows users to choose the `color of the overlay boxes` around detected enemies.
- 🛠️ **GPU Acceleration**: Supports `GPU acceleration` for faster processing with `CUDA-enabled GPUs`.
- 🎥 **Live Feed Support**: Displays a real-time live feed with object detection overlays.
- 🖥️ **AMD GPU with DirectML** support (for faster processing).

## 🖥️ System Requirements

- **Operating System**: Windows
- **Python Version**: [Python 3.11.6](https://github.com/KernFerm/Py3.11.6installer) Click to download
- **Hardware**:
  - **CPU**: Multi-core processor (Intel i5 or Better)
  - **GPU** (Optional, but recommended for better performance): `NVIDIA GPU with CUDA support`
  - **RAM**: 16 GB or more (games you have on your PC recommended RAM 16GB)
- **Command Prompt** - Comes standard with `ALL` windows computers. 
  - click `start menu` in the `search bar` type `cmd.exe` click enter. :) 
  - to make sure you have installed `Python 3.11.6` correctly:
    - type in `cmd.exe`

    ```
    python --version 
    ```
- Use the `install_python.bat` it will auto add to `system path`. `V3.11.6`

## 📦 Installation

Follow these steps to set up and run **GameVisionAid**:

1. **Clone the repository**:
    ```
    git clone https://github.com/kernferm/game-vision-aid.git
    cd game-vision-aid
    ```

2. **Set up a virtual environment (optional but recommended)**:
    ```
    python -m venv game-vision-aid_env
    ```

   To deactivate the virtual environment:
    ```
    deactivate
    ```

   On Windows, activate the environment:
    ```
    game-vision-aid_env\Scripts\activate
    ```

3. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

4. **For NVIDIA GPU Support (Optional)**:
   - If you have a CUDA-enabled GPU, install the version of PyTorch that supports your CUDA version:
     ```
     pip3 install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
     ```

5. **For CPU Only** (Laptops with no GPU):
   - Use the `install_pytorch_cpu_only.bat` to install CPU-based PyTorch.
   - If `install_pytorch_cpu_only.bat` doesn't work, you can manually run the following command in `CMD.exe`:
     ```
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu/torch_stable.html
     ```

---

## 💻 For `AMD GPU` Users

If you have an **AMD GPU**, follow the steps below to set up DirectML support for faster processing:

1. **Install AMD GPU dependencies**:
   - Run the **`amd_gpu_requirements.bat`** script to install the necessary dependencies for AMD GPU:
     ```
     amd_gpu_requirements.bat
     ```

   This will install **DirectML** and other required packages, enabling the project to run efficiently on AMD hardware.

2. **Launch the application**:
   - After installing the dependencies, use the **`amdgpu-launcher.bat`** script to run the application with AMD GPU support.

3. Check **## How to Use config.py** 
---

### Troubleshooting

If you encounter any issues, such as an **Ultralytics error**, follow the steps below:

1. Run the following command in `CMD.exe` to upgrade Ultralytics:
    ```
    pip install --upgrade ultralytics
    ```

2. The `Ultralytics` package is already included in the `requirements.txt` and `requirements.bat` files.

3. Use the `update_ultralytics.bat` script if you continue to experience Ultralytics errors.

### Note 
  - If you get an Ultralytics error when installing 
  Run the Command below in `CMD.exe`
    ```
    pip install --upgrade ultralytics
    ```
  - I did include the `Ultralytics` in the requirements.txt` and `requirements.bat`.
  - Use the `install_pytorch.bat` as it is tied to `cu118`. (Recommended)
  - Use the `update_ultralytics.bat` if you have `ultralytics error`


# 🚀 NVIDIA CUDA Installation Guide

### 1. **Download the NVIDIA CUDA Toolkit 11.8**

First, download the CUDA Toolkit 11.8 from the official NVIDIA website:

👉 [Nvidia CUDA Toolkit 11.8 - DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. **Install the CUDA Toolkit**

- After downloading, open the installer (`.exe`) and follow the instructions provided by the installer.
- Make sure to select the following components during installation:
  - CUDA Toolkit
  - CUDA Samples
  - CUDA Documentation (optional)

### 3. **Verify the Installation**

- After the installation completes, open the `cmd.exe` terminal and run the following command to ensure that CUDA has been installed correctly:
  ```
  nvcc --version
  ```
This will display the installed CUDA version.

### **4. Install Cupy**
Run the following command in your terminal to install Cupy:
  ```
  pip install cupy-cuda11x
  ```

```
@echo off
echo MAKE SURE TO HAVE THE WHL DOWNLOADED BEFORE YOU CONTINUE!!!
pause
echo Click the link to download the WHL: press ctrl then left click with mouse
echo https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo Installing CuPy from WHL...
pip install https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo All packages installed successfully!
pause
```



## 5. CUDNN Installation 🧩
Download cuDNN (CUDA Deep Neural Network library) from the NVIDIA website:

👉 [Download CUDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). (Requires an NVIDIA account – it's free).

## 6. Unzip and Relocate 📁➡️
Open the `.zip` cuDNN file and move all the folders/files to the location where the CUDA Toolkit is installed on your machine, typically:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 7. Get TensorRT 8.6 GA 🔽
Download [TensorRT 8.6 GA](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

## 8. Unzip and Relocate 📁➡️
Open the `.zip` TensorRT file and move all the folders/files to the CUDA Toolkit folder, typically located at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 9. Python TensorRT Installation 🎡
Once all the files are copied, run the following command to install TensorRT for Python:

```
pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

🚨 **Note:** If this step doesn’t work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Just locate the correct `.whl` file in the `python` folder and replace the path accordingly.

## 10. Set Your Environment Variables 🌎
Add the following paths to your environment variables:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

# Setting Up CUDA 11.8 with cuDNN on Windows

Once you have CUDA 11.8 installed and cuDNN properly configured, you need to set up your environment via `cmd.exe` to ensure that the system uses the correct version of CUDA (especially if multiple CUDA versions are installed).

## Steps to Set Up CUDA 11.8 Using `cmd.exe`

### 1. Set the CUDA Path in `cmd.exe`

You need to add the CUDA 11.8 binaries to the environment variables in the current `cmd.exe` session.

Open `cmd.exe` and run the following commands:

```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%
```
These commands add the CUDA 11.8 binary, lib, and CUPTI paths to your system's current session. Adjust the paths as necessary depending on your installation directory.

2. Verify the CUDA Version
After setting the paths, you can verify that your system is using CUDA 11.8 by running:
```
nvcc --version
```
This should display the details of CUDA 11.8. If it shows a different version, check the paths and ensure the proper version is set.

3. **Set the Environment Variables for a Persistent Session**
If you want to ensure CUDA 11.8 is used every time you open `cmd.exe`, you can add these paths to your system environment variables permanently:

1. Open `Control Panel` -> `System` -> `Advanced System Settings`.
Click on `Environment Variables`.
Under `System variables`, select `Path` and click `Edit`.
Add the following entries at the top of the list:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
```
This ensures that CUDA 11.8 is prioritized when running CUDA applications, even on systems with multiple CUDA versions.

4. **Set CUDA Environment Variables for cuDNN**
If you're using cuDNN, ensure the `cudnn64_8.dll` is also in your system path:
```
set PATH=C:\tools\cuda\bin;%PATH%
```
This should properly set up CUDA 11.8 to be used for your projects via `cmd.exe`.

#### Additional Information
- Ensure that your GPU drivers are up to date.
- You can check CUDA compatibility with other software (e.g., PyTorch or TensorFlow) by referring to their documentation for specific versions supported by CUDA 11.8.

---

## ⚙️ Configuration: `config.py`
 - **Remember to `SAVE` your `config.py` or from the `config-launcher.bat` settings then re start `## Usage Section` on the `readme.md`
The `config.py` file allows you to easily configure screen capture, object detection, YOLO model settings, and bounding box colors for your specific needs.

```
# Configuration for BetterCam Screen Capture and YOLO model

# Screen Capture Settings
screenWidth = 480  # Updated screen width for better resolution
screenHeight = 480  # Updated screen height for better resolution

# Object Detection Settings
confidenceThreshold = 0.5  # Confidence threshold for object detection
nmsThreshold = 0.4  # Non-max suppression threshold to filter overlapping boxes

# YOLO Model Settings
modelType = 'onnx'  # Choose 'torch' or 'onnx' based on the model you want to load
torchModelPath = 'models/fn_v5.pt'  # Path to YOLOv5 PyTorch model
onnxModelPath = 'models/fn_v5.onnx'  # Path to YOLOv5 ONNX model

# BetterCam Settings
targetFPS = 60  # Frames per second for capturing
maxBufferLen = 512  # Max buffer length for storing frames
region = None  # Region for capture (set to None for full screen)
useNvidiaGPU = True  # Set to True to enable GPU acceleration if available

# Colors for Bounding Boxes
boundingBoxColor = (0, 255, 0)  # Default bounding box color in BGR format
highlightColor = (0, 0, 255)  # Color for highlighted objects
```

## How to Use config.py
- **Remember to `SAVE` your `config.py` or from the `config-launcher.bat` settings then re start `## Usage Section` on the `readme.md`
1. Open the `config.py` file.

2. Set the `screenWidth` and `screenHeight` to match your preferred screen resolution.

3. Adjust `confidenceThreshold` to modify the object detection sensitivity (default is set to `0.5` for moderate confidence).

4. Configure the `YOLO Model Settings` to specify whether you want to use the `PyTorch` or `ONNX` model.

5. Set the desired `targetFPS`, and if you have an NVIDIA GPU, ensure `useNvidiaGPU` is set to `True` for optimal performance.

6. Customize the `boundingBoxColor` and `highlightColor` to match your preference for object detection overlays.

7. Save the changes, and the settings will automatically be applied when you run the program

- Use the `config-launcher.bat` to open the `config.py`
- Make `sure` to run the `config-launcher.bat` in the same folder as the `config.py` 

# Color Dictionary

This color dictionary includes a range of colors that are **high-contrast**, **colorblind-friendly**, and **visually impaired-friendly**. These colors have been selected to provide maximum accessibility and usability for a broad range of users.

## High-Contrast, Colorblind-Friendly and Visually Impaired-Friendly Colors

| Color        | RGB Values        | Notes                                                |
|--------------|-------------------|------------------------------------------------------|
| **Red**      | (0, 0, 255)       | Good contrast with green, blue, yellow                |
| **Green**    | (0, 255, 0)       | High contrast with red, magenta                       |
| **Blue**     | (255, 0, 0)       | Standard blue                                        |
| **Yellow**   | (0, 255, 255)     | High contrast, suitable for colorblindness            |
| **Cyan**     | (255, 255, 0)     | Good contrast with red, blue                         |
| **Magenta**  | (255, 0, 255)     | High contrast                                        |
| **Black**    | (0, 0, 0)         | High contrast with all light colors                  |
| **White**    | (255, 255, 255)   | High contrast with dark colors                       |

## Colorblind-Friendly and High Contrast Shades

| Color            | RGB Values        | Notes                                                |
|------------------|-------------------|------------------------------------------------------|
| **Dark Red**     | (139, 0, 0)       | Distinguishable from green, easier for colorblind users |
| **Orange**       | (0, 165, 255)     | Strong contrast with most colors                     |
| **Light Blue**   | (173, 216, 230)   | Easily distinguishable from green                    |
| **Purple**       | (128, 0, 128)     | High contrast, distinguishable for most users        |
| **Brown**        | (165, 42, 42)     | Distinct and neutral color                           |
| **Pink**         | (255, 182, 193)   | Good contrast, especially for colorblindness         |
| **Lime**         | (0, 255, 128)     | Bright color, good contrast with red and blue        |
| **Turquoise**    | (64, 224, 208)    | Strong contrast with most shades                     |
| **Navy**         | (0, 0, 128)       | High contrast for visual impairment                  |
| **Gold**         | (255, 215, 0)     | Bright and distinguishable                           |
| **Silver**       | (192, 192, 192)   | Neutral, high contrast with darker colors            |
| **Dark Orange**  | (255, 140, 0)     | High contrast and distinguishable                    |
| **Indigo**       | (75, 0, 130)      | Deep color, good contrast with light shades          |
| **Teal**         | (0, 128, 128)     | Strong contrast, good for visual impairments         |
| **Olive**        | (128, 128, 0)     | Neutral, good for colorblind users                   |
| **Maroon**       | (128, 0, 0)       | Distinct and high contrast                           |
| **Sky Blue**     | (135, 206, 235)   | Bright and easily visible                            |
| **Chartreuse**   | (127, 255, 0)     | High contrast for most users                         |



### 🖼️ Region Capture

- **Region for Capture** (`config.region` in `config.py`): This setting defines the area of the screen that BetterCam will capture for object detection.
  - `None`: Captures the entire screen.
  - `(x1, y1, x2, y2)`: Captures a specific portion of the screen defined by the coordinates `(x1, y1)` as the top-left corner and `(x2, y2)` as the bottom-right corner (e.g., `(100, 100, 800, 600)` will capture a section from (100,100) to (800,600)).
  
By adjusting this setting, you can focus the capture on a particular region of the screen, which can help reduce processing load and ignore unnecessary areas. It’s especially useful for games or applications where you only need to detect objects in a specific portion of the display.


## 📝 Usage

1. **Run the GameVisionAid script:**
- Open up `CMD.exe` non admin mode , copy the file location 
- `right click` on file `properties` `location` copy the entire location
- go back to `CMD.exe` 
- type `cd` 
- `paste` the location you just `copied` to `CMD.exe`
- `remove` the part where it says `main.py` at the end of the `file location` in `CMD.exe`
- then `press enter` 
- copy and paste `code below`
- then `enter`
```
python main.py
```
#### Or Use Below for Easy: 
- Use the `main-launcher.bat` to run the `main.py`
- Make `sure` to run the `main-launcher.bat` in the same folder of the `main.py`

2. **Select the overlay color:**

A prompt will appear. Enter the color name you want for the overlay boxes around detected enemies. You can type `exit` or `q` to quit the program.

3. **Start your game:**

The overlay will run in parallel with your game, highlighting detected enemies with the chosen color.

4. **Exit the overlay or the Program:**

Press `q` or type `exit` at any time "after you've start the program of course" to close the overlay and exit the program.


## 🤖 Custom YOLOv5 Model (Optional)

- If you have a custom-trained YOLOv5 model specifically for your game:

1. Place your `.pt` or `.onnx` file in the models/ directory.

2. Update the `config.py` file to load your custom model by setting:

```
torchModelPath = 'models/your_model.pt'  # or
onnxModelPath = 'models/your_model.onnx'
```

#### ❗ MAKE SURE TO UNZIP THE models.zip in the same directory as the main.py script.


## ❗ Known Issues

- **Performance on CPU:** The overlay may run slower on systems without a GPU. Using a CUDA-enabled GPU is recommended for real-time performance.

- **Compatibility:** This tool is intended for games that allow screen capturing and do not violate any terms of service.

## 🛠️ Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements, bugs, or new features.
```
## want to contribute

### please read the repo

- use python 3.11.6
- fork the repo if you want to contribute ONLY!!!
- make a pull request
- there are 5 branches - use `patch-1` to upload or `patch-2` upload new code. 
- download the `main-branch`
- trying to get the live feed to work properly - (shows up multiple times, i will double check)
- trying to get the visual colors to work properly 

#### if you find anything else, make a notes-2.txt.
```
## 📜 License

**This project is proprietary, and all rights are reserved by the author. Unauthorized copying, distribution, or modification of this project is strictly prohibited unless you have written permission from the developer or the FNBUBBLES420 ORG.**

## 📧 Contact

- **[Bubbles The Dev](https://github.com/FNBUBBLES420-ORG/game-vision-aid/blob/main/Contact.md)**
- **[Main Office](https://github.com/FNBUBBLES420-ORG/game-vision-aid/blob/main/Contact.md)**

## 🙏 Acknowledgements

Thanks to the developers of:

- **[Bettercam](https://github.com/RootKit-Org/BetterCam)**

- **[Yolo v5](https://github.com/ultralytics/yolov5)**

## 👨‍💻 Developed by [Bubbles The Dev](https://github.com/kernferm) - Making gaming more accessible for everyone!

## ⚠️ Disclaimer

`GameVisionAid` **is designed to assist visually impaired or color-blind gamers by enhancing visual cues in video games. It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.**

**This tool runs in parallel with any game and does not modify the game files or violate any terms of service. It captures the screen and provides an overlay to help users better perceive in-game elements. The developer is not responsible for any misuse of this tool.**
