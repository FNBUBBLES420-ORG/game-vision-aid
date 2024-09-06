
![Game Vision Aid Banner](https://github.com/FNBUBBLES420-ORG/game-vision-aid/blob/main/banner/Game_Vision_Aid.png)




# 🎮 GameVisionAid

**GameVisionAid** is an accessibility tool designed to assist visually impaired gamers by enhancing visual cues in video games. It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.

## 🚀 Features

- 🖥️ **Real-Time Screen Capture**: Captures your screen in real-time using `BetterCam` for fast and efficient screen capturing.
- 🎯 **Object Detection**: Utilizes `YOLOv5` for detecting enemies in video games.
- 🟩 **Customizable Overlays**: Allows users to choose the `color of the overlay boxes` around detected enemies.
- 🛠️ **GPU Acceleration**: Supports `GPU acceleration` for faster processing with `CUDA-enabled GPUs`.
- 🎥 **Live Feed Support**: Displays a real-time live feed with object detection overlays.

## 🖥️ System Requirements

- **Operating System**: Windows
- **Python Version**: [Python 3.11.6](https://github.com/KernFerm/Py3.11.6installer)
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

   On Windows use:
    ```
    game-vision-aid_env\Scripts\activate
    ```

3. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

4. **For GPU Support (Optional)**:
   - If you have a CUDA-enabled GPU, install the version of PyTorch that supports your CUDA version:
     ```
     pip3 install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
     ```

5. **For CPU Only** aka laptops with no `GPU`
   - Use the `install_pytorch_cpu_only.bat` to install `CPU` Pytorch

### Note 
  - If you get an Ultralytics error when installing 
  Run the Command below in `CMD.exe`
    ```
    pip install --upgrade ultralytics
    ```
  - I did include the `Ultralytics` in the requirements.txt` and `requirements.bat`.
  - Use the `install_pytorch.bat` as it is tied to `cu118`. (Recommended)
  - Use the `update_ultralytics.bat` if you have `ultralytics error`


5. **CUDA & cuDNN**:
   - You will have to **Download** and **Install** CUDA & cuDNN manually and extract the zip file.
   - You will have to create a **Nvidia Developer Account**.
   - Make sure to have **Node.js** installed from [Node JS](https://nodejs.org/dist/v20.17.0/node-v20.17.0-x64.msi).
   - Run the `nodejs-instructions.ps1` in **NON ADMIN**.
   - [cuDNN 8.9.7](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/11.x/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip/):
     ```
     node cudnn_instruction.js
     ```
   - [CUDA 11.8.0](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe).

## ⚙️ Configuration: `config.py`

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
1. Open the `config.py` file.

2. Set the `screenWidth` and `screenHeight` to match your preferred screen resolution.

3. Adjust `confidenceThreshold` to modify the object detection sensitivity (default is set to `0.5` for moderate confidence).

4. Configure the `YOLO Model Settings` to specify whether you want to use the `PyTorch` or `ONNX` model.

5. Set the desired `targetFPS`, and if you have an NVIDIA GPU, ensure `useNvidiaGPU` is set to `True` for optimal performance.

6. Customize the `boundingBoxColor` and `highlightColor` to match your preference for object detection overlays.

7. Save the changes, and the settings will automatically be applied when you run the program

- Use the `config-launcher.bat` to open the `config.py`
- Make `sure` to run the `config-launcher.bat` in the same folder as the `config.py` 

### 🖼️ Region Capture

- **Region for Capture** (`config.region` in `config.py`): This setting defines the area of the screen that BetterCam will capture for object detection.
  - `None`: Captures the entire screen.
  - `(x1, y1, x2, y2)`: Captures a specific portion of the screen defined by the coordinates `(x1, y1)` as the top-left corner and `(x2, y2)` as the bottom-right corner (e.g., `(100, 100, 800, 600)` will capture a section from (100,100) to (800,600)).
  
By adjusting this setting, you can focus the capture on a particular region of the screen, which can help reduce processing load and ignore unnecessary areas. It’s especially useful for games or applications where you only need to detect objects in a specific portion of the display.


## 📝 Usage

1. **Run the GameVisionAid script:**

```
python main.py
```
- Use the `main-launcher.bat` to run the `main.py`
- Make `sure` to run the `main-launcher.bat` in the same folder of the `main.py`

2. **Select the overlay color:**

A prompt will appear. Enter the color name you want for the overlay boxes around detected enemies. You can type `exit` or `q` to quit the program.

3. **Start your game:**

The overlay will run in parallel with your game, highlighting detected enemies with the chosen color.

4. **Exit the overlay or the Program:**

Press `q` or type `exit` at any time "after you've start the program of course" to close the overlay and exit the program.

## 📂 Project Structure

```
game-vision-aid/ 
├── .github/                              # For Issues
├── banner folder                         # Banner Image PNG
├── models/                               # Where the models go
│   └── fn_v5.pt                          # Custom YOLOv5 model (if available)
│    └── model_v5s.pt                     # Custom YOLOv5 model (if available)
│    └── fn_v5v5480480Half.onnx           # Custom ONNX model (if available) nvidia gpu 2060 super
│    └── model_fn_v5v5320320Half.onnx     # Custom ONNX model (if available) nvidia gpu 2060 super
│    └── model_v5sv5320320Half.onnx       # Custom ONNX model (if available) nvidia gpu 2060 super
│    └── model_v5sv5480480Half.onnx       # Custom ONNX model (if available) nvidia gpu 2060 super
├── src/                                  # Part of RUST Application
│   └── main.rs                           # Part of RUST Application
├── .gitignore                            # gitignore
├── CODE_OF_CONDUCT.md                    # CODE_OF_CONDUCT.md
├── Cargo.lock                            # Part of RUST Application
├── Cargo.toml                            # Part of RUST Application
├── LICENSE                               # LICENSE
├── README.md                             # Project documentation
├── SECURITY.md                           # Security documentation
├── config-launcher.bat                   # config-launcher batchfile to launches config.py in the root directory of folder
├── config.py                             # Configuration file 
├── cudnn_instructions.js                 # Instructions in JavaScript
├── install_pytorch.bat                   # Pytorch Build wheel for the script
├── main-launcher.bat                     # main-launcher batchfile launches the main.py in the root directory of folder
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

## 📜 License

**This project is proprietary, and all rights are reserved by the author. Unauthorized copying, distribution, or modification of this project is strictly prohibited unless you have written permission from the developer or the FNBUBBLES420 ORG.**

## 📧 Contact

- **[Bubbles The Dev](kernferm@gmail.com)**
- **[Main Office](mainoffice@fnbubbles420.org)**

## 🙏 Acknowledgements

Thanks to the developers of:

- **[Bettercam](https://github.com/RootKit-Org/BetterCam)**

- **[Yolo v5](https://github.com/ultralytics/yolov5)**

## 👨‍💻 Developed by [Bubbles The Dev](https://github.com/kernferm) - Making gaming more accessible for everyone!

## ⚠️ Disclaimer

`GameVisionAid` **is designed to assist visually impaired or color-blind gamers by enhancing visual cues in video games. It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.**

**This tool runs in parallel with any game and does not modify the game files or violate any terms of service. It captures the screen and provides an overlay to help users better perceive in-game elements. The developer is not responsible for any misuse of this tool.**
