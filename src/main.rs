
    fn main() {
        // 🎮 GameVisionAid
        println!("🎮 GameVisionAid:");
        println!("GameVisionAid is an accessibility tool designed to assist visually impaired gamers by enhancing visual cues in video games.");
        println!("It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.\n");
    
        // 🚀 Features
        println!("🚀 Features:");
        println!("🖥️ Real-Time Screen Capture: Captures your screen in real-time using `BetterCam` for fast and efficient screen capturing.");
        println!("🎯 Object Detection: Utilizes YOLOv5 for detecting enemies in video games.");
        println!("🟩 Customizable Overlays: Allows users to choose the color of the overlay boxes around detected enemies.");
        println!("🛠️ GPU Acceleration: Supports GPU acceleration for faster processing with CUDA-enabled GPUs.");
        println!("🎥 Live Feed Support: Displays a real-time live feed with object detection overlays.");
        println!("🖥️ AMD GPU with DirectML support (for faster processing).\n");
    
        // 🖥️ System Requirements
        println!("🖥️ System Requirements:");
        println!("- Operating System: Windows");
        println!("- Python Version: Python 3.11.6 (Click to download)");
        println!("- Hardware:");
        println!("  - CPU: Multi-core processor (Intel i5 or Better)");
        println!("  - GPU (Optional, but recommended for better performance): NVIDIA GPU with CUDA support");
        println!("  - RAM: 16 GB or more (games you have on your PC recommended RAM 16GB)");
        println!("- Command Prompt: Comes standard with ALL windows computers.");
        println!("To make sure you have installed Python 3.11.6 correctly:");
        println!("Type in `cmd.exe`:");
        println!("python --version");
        println!("Use the `install_python.bat` to auto-add to system path. V3.11.6\n");
    
        // 📦 Installation
        println!("📦 Installation:");
        println!("Follow these steps to set up and run GameVisionAid:");
        println!("1. Clone the repository:");
        println!("   git clone https://github.com/kernferm/game-vision-aid.git");
        println!("   cd game-vision-aid");
        println!("2. Set up a virtual environment (optional but recommended):");
        println!("   python -m venv game-vision-aid_env");
        println!("3. To deactivate the virtual environment:");
        println!("   deactivate");
        println!("4. On Windows, activate the environment:");
        println!("   game-vision-aid_env\\Scripts\\activate\n");
    
        println!("5. Install dependencies:\n");
        println!("pip install -r requirements.txt\n");
    
        // NVIDIA GPU Support
        println!("For NVIDIA GPU Support (Optional):");
        println!("If you have a CUDA-enabled GPU, install the version of PyTorch that supports your CUDA version:");
        println!("pip3 install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118\n");
    
        // For CPU only
        println!("For CPU Only (Laptops with no GPU):");
        println!("Use the `install_pytorch_cpu_only.bat` to install CPU-based PyTorch.");
        println!("If `install_pytorch_cpu_only.bat` doesn't work, run the following in `CMD.exe`:");
        println!("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu/torch_stable.html\n");
    
        // For AMD GPU users
        println!("💻 For AMD GPU Users:");
        println!("If you have an AMD GPU, follow the steps below to set up DirectML support for faster processing:");
        println!("1. Install AMD GPU dependencies:");
        println!("   Run the `amd_gpu_requirements.bat` script to install the necessary dependencies for AMD GPU:");
        println!("   amd_gpu_requirements.bat");
        println!("2. Launch the application:");
        println!("   After installing the dependencies, use the `amdgpu-launcher.bat` script to run the application with AMD GPU support.\n");
        
        // Troubleshooting
        println!("🛠️ Troubleshooting:");
        println!("If you encounter any issues, such as an Ultralytics error:");
        println!("Run the following command in `CMD.exe` to upgrade Ultralytics:");
        println!("pip install --upgrade ultralytics\n");
    
        // Configuration
        println!("⚙️ Configuration: config.py");
        println!("The `config.py` file allows you to easily configure screen capture, object detection, YOLO model settings, and bounding box colors for your specific needs.");
        println!("Make sure to SAVE your `config.py` or from the `config-launcher.bat` settings and restart.\n");
    
        println!("📝 How to Use config.py:");
        println!("1. Open the `config.py` file.");
        println!("2. Set the `screenWidth` and `screenHeight` to match your preferred screen resolution.");
        println!("3. Adjust `confidenceThreshold` to modify object detection sensitivity.");
        println!("4. Configure the YOLO Model Settings to use either the PyTorch or ONNX model.");
        println!("5. Set `targetFPS` and if you have an NVIDIA GPU, ensure `useNvidiaGPU` is set to `True` for optimal performance.");
        println!("6. Customize the `boundingBoxColor` and `highlightColor`.");
        println!("7. Save changes to `config.py`. The settings will be applied when you run the program.");
        println!("8. Use the `config-launcher.bat` to open and edit `config.py`.\n");
    
        // 🖼️ Region Capture
        println!("🖼️ Region Capture:");
        println!("Region for Capture (`config.region` in `config.py`): This setting defines the area of the screen that BetterCam will capture for object detection.");
        println!("None: Captures the entire screen.");
        println!("(x1, y1, x2, y2): Captures a specific portion of the screen defined by the coordinates.");
        println!("By adjusting this setting, you can reduce processing load by focusing on specific parts of the screen.\n");
    
        // 📝 Usage Instructions
        println!("📝 Usage:");
        println!("1. Open `CMD.exe` in non-admin mode, copy the file location.");
        println!("2. Right-click on file properties, copy the location.");
        println!("3. Paste the location in `cmd.exe`, remove `main.py`, and press enter.");
        println!("4. Run the command:");
        println!("   python main.py");
        println!("Or, use `main-launcher.bat` to run `main.py`.\n");
    
        // Exiting the Program
        println!("Exit the overlay by pressing 'q' or typing 'exit' after starting the program.\n");
    
        // Project Structure
        println!("📂 Project Structure:");
        println!("game-vision-aid/");
        println!("├── .github/                              # For Issues");
        println!("├── AMD-GPU-Only folder                   # AMD GPU folder");
        println!("├──    └── amd_gpu_requirements.bat       # AMD GPU requirements batchfile");
        println!("├──    └── amdgpu-launcher.bat            # AMD GPU launcher batchfile");
        println!("├──    └── config-launcher.bat            # AMD config-launcher batchfile");
        println!("├──     └── config.py                     # AMD Python Configuration file");
        println!("├──     └── main.py                       # AMD Main script file");
        println!("├──     └── requirements.txt              # AMD List of required Python packages");
        println!("├── banner folder                         # Banner Image PNG");
        println!("├── models/                               # Where the models go");
        println!("│   └── fn_v5.pt                          # Custom YOLOv5 model (if available)");
        println!("│    └── model_v5s.pt                     # Custom YOLOv5 model (if available)");
        println!("│    └── fn_v5v5480480Half.onnx           # Custom ONNX model (if available) nvidia gpu 2060 super");
        println!("│    └── model_fn_v5v5320320Half.onnx     # Custom ONNX model (if available) nvidia gpu 2060 super");
        println!("│    └── model_v5sv5320320Half.onnx       # Custom ONNX model (if available) nvidia gpu 2060 super");
        println!("│    └── model_v5sv5480480Half.onnx       # Custom ONNX model (if available) nvidia gpu 2060 super");
        println!("├── src/                                  # Part of RUST Application");
        println!("│   └── main.rs                           # Part of RUST Application");
        println!("├── .gitignore                            # gitignore");
        println!("├── CODE_OF_CONDUCT.md                    # CODE_OF_CONDUCT.md");
        println!("├── Cargo.lock                            # Part of RUST Application");
        println!("├── Cargo.toml                            # Part of RUST Application");
        println!("├── LICENSE                               # LICENSE");
        println!("├── README.md                             # Project documentation");
        println!("├── SECURITY.md                           # Security documentation");
        println!("├── config-launcher.bat                   # config-launcher batchfile to launches config.py in the root directory of folder");
        println!("├── config.py                             # Configuration file");
        println!("├── cudnn_instructions.js                 # Instructions in JavaScript");
        println!("├── install_python.bat                    # Python 3.11.6 Batchfile");
        println!("├── install_pytorch.bat                   # Pytorch Build wheel for the script GPU");
        println!("├── install_pytorch_cpu_only.bat          # Pytorch Build for CPU");
        println!("├── main-launcher.bat                     # main-launcher batchfile launches the main.py in the root directory of folder");
        println!("├── main.py                               # Main script file");
        println!("├── model_v5s.pt                          # pt file goes in modelS folder");
        println!("├── model_v5sv5480480Half.onnx            # onnx file goes in modelS folder");
        println!("├── model_fn_v5v5320320Half.onnx          # onnx file goes in modelS folder");
        println!("├── models-fn_v5.zip                      # Compressed models to in main models folder");
        println!("├── nodejs-instructions.ps1               # Instructions in Powershell");
        println!("├── notes.txt                             # Read Notes.txt");
        println!("├── requirements.bat                      # Installs list of required Python packages");
        println!("├── requirements.txt                      # List of required Python packages");
        println!("├── update_ultralytics.bat                # Update Batch Script for Ultralytics\n");
    
        // Custom YOLOv5 Model
        println!("🤖 Custom YOLOv5 Model (Optional):");
        println!("If you have a custom-trained YOLOv5 model for your game, place your `.pt` or `.onnx` file in the `models/` directory.");
        println!("Update `config.py` to load your custom model by setting `torchModelPath` or `onnxModelPath`.\n");
    
        // Known Issues
        println!("❗ Known Issues:");
        println!("Performance on CPU: The overlay may run slower on systems without a GPU. Using a CUDA-enabled GPU is recommended for real-time performance.");
        println!("Compatibility: This tool is intended for games that allow screen capturing and do not violate any terms of service.\n");
    
        // Contributing
        println!("🛠️ Contributing:");
        println!("Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements, bugs, or new features.\n");
    
        // License
        println!("📜 License:");
        println!("This project is proprietary, and all rights are reserved by the author. Unauthorized copying, distribution, or modification is prohibited.\n");
    
        // Contact
        println!("📧 Contact:");
        println!("Bubbles The Dev - Main Office\n");
    
        // Acknowledgements
        println!("🙏 Acknowledgements:");
        println!("Thanks to the developers of BetterCam, YOLOv5, and the FNBUBBLES420 ORG team.\n");
    
        // Disclaimer
        println!("⚠️ Disclaimer:");
        println!("GameVisionAid is designed to assist visually impaired or color-blind gamers by enhancing visual cues. It runs in parallel with any game without modifying game files or violating terms of service.\n");
   
    
        // Supported Pytorch and Torchvision versions
        println!("Supported Pytorch and Torchvision versions:");
        println!("torch==2.4.1");
        println!("torchvision==0.19.1\n");

        // Cuda and CuDNN versions
        println!("CUDA and cuDNN versions:");
        println!("CUDA Version: 11.8");
        println!("cuDNN Version: 8.6.9\n");

        // Python Version
        println!("Python Version:");
        println!("Version: 3.11.6\n");

        // Programming Languages Used
        println!("Programming Languages Used:");
        println!("1. Python");
        println!("2. JavaScript");
        println!("3. Rust");
        println!("4. Batchfile");
        println!("5. PowerShell\n");

        // Nvidia Dependencies
        println!("Nvidia Dependencies:");
        println!("1. PyTorch");
        println!("2. Torchvision");
        println!("3. OpenCV");
        println!("4. NumPy");
        println!("5. CUDA");
        println!("6. cuDNN");
        println!("7. GitPython");
        println!("8. BetterCAM");
        println!("9. Pandas");
        println!("10. Colorama");
        println!("11. OnnxRuntime");
        println!("12. Requests\n");

        // AMD Dependencies
        println!("AMD Dependencies:");
        println!("1. opencv-python");
        println!("2. torch");
        println!("3. torchvision");
        println!("4. torchaudio");
        println!("5. numpy");
        println!("6. gitpython==3.1.43");
        println!("7. pandas");
        println!("8. colorama");
        println!("9. onnxruntime");
        println!("10. requests");
        println!("11. bettercam");
        println!("12. torch-directml");
        println!("13. onnxruntime-directml\n");

        // Dependencies Versions
        println!("Dependencies Versions:");
        println!("BetterCam Version: BetterCam 1.0");
        println!("PyTorch Version: 2.4.1");
        println!("Torchvision Version: 0.19.1");
        println!("Torch Audio Version: 2.4.1");
        println!("OpenCV Version: 4.9.0.80");
        println!("NumPy Version: 1.26.4");
        println!("GitPython Version: 3.1.43");
        println!("Pandas Version: 2.2.2");
        println!("Colorama Version: 0.4.6");
        println!("OnnxRuntime Version: 1.19.2");
        println!("Requests Version: 2.32.3");
        println!("Ultralytics Version: 8.2.89");
        println!("OnnxRuntime-DirectML Version: 1.9.2\n");

        // Os Supported
        println!("OS Supported:");
        println!("1. Windows 10");
        println!("2. Windows 11\n");

        // Game Vision Aid Version
        println!("Game Vision Aid Version:");
        println!("Version: 1.1.48\n");

        // AMD GPU Support
        println!("AMD GPU:");
        println!("AMD GPU Support: Supported\n");

        // AMD CPU Support
        println!("AMD CPU:");
        println!("AMD CPU Support: Supported\n");

        // Intel CPU Support
        println!("Intel CPU:");
        println!("Intel CPU Support: Supported\n");

        // NVIDIA GPU Support
        println!("NVIDIA:");
        println!("NVIDIA GPU Support: Supported");
        println!("NVIDIA GPU: NVIDIA GeForce RTX 2060 Super - Supported\n");
        println!("NVIDIA GPU with CUDA support");
        println!("CUDA Toolkit 11.8");
        println!("cuDNN 8.9.6\n");

        // Intel GPU Support
        println!("Intel GPU:");
        println!("Intel GPU Support: NOT Supported\n");

        // Object Detection Model
        println!("Object Detection:");
        println!("Object Detection Model: YOLOv5\n");

        // CPU Support
        println!("CPU Support:");
        println!("CPU Support: Multi-core processor (AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz or or better)");
        println!("CPU Support: Intel(R) Core(TM) i7-14700F   2.10 GHz \n");

        // Yolo version
        println!("YOLO Version Name:");
        println!("YOLO Version: YOLOv5\n");

        // CUDA Version
        println!("CUDA File Name:");
        println!("CUDA Version: 11.8\n");

        // cuDNN Version
        println!("cuDNN File Name:");
        println!("cuDNN Version: 8.9.6\n");

        // Batchfile File Names
        println!("Batchfile File Names:");
        println!("1. requirements.bat");
        println!("2. install_pytorch.bat");
        println!("3. update_ultralytics.bat");
        println!("4. install_pytorch_cpu_only.bat");
        println!("5. main-launcher.bat");
        println!("6. config-launcher.bat");
        println!("7. install_python.bat");
        println!("8. amdgpu-launcher.bat");
        println!("9. amd_gpu_requirements.bat");
        println!("10. config-launcher.bat\n");

        // PowerShell File Names
        println!("PowerShell File Names:");
        println!("1. nodejs-instructions.ps1\n");

        // Rust File Names
        println!("Rust File Names:");
        println!("1. main.rs\n");

        // JavaScript File Names
        println!("JavaScript File Names:");
        println!("1. cudnn_instruction.js\n");

        // Python File Names
        println!("Python File Names:");
        println!("1. config.py");
        println!("2. main.py\n");

        // MD File Names
        println!("MD File Names:");
        println!("1. README.md");
        println!("2. CONTRIBUTING.md");
        println!("3. LICENSE.md");
        println!("4. CODE_OF_CONDUCT.md");
        println!("5. SECURITY.md\n");

        // Txt File Names
        println!("Txt File Names:");
        println!("1. requirements.txt\n");

        // PNG File Names
        println!("PNG File Names:");
        println!("1. Game_Vision_Aid.png\n");
    
        // Usage
        println!("Usage:");
        println!("After building the project, you can run the executable to start using the Game Vision Aid tools:");
        println!("./target/release/game-vision-aid\n");

        // JavaScript Code Explanation
        println!("JavaScript Code Explanation:");
        println!("console.log(\"cuDNN is not installed. Please follow these steps to download and install cuDNN 8.9.6 manually:\");");
        println!("console.log(\"1. Go to the NVIDIA cuDNN download page: https://developer.nvidia.com/rdp/cudnn-download\");");
        println!("console.log(\"2. Log in with your NVIDIA Developer account.\");");
        println!("console.log(\"3. Download the file 'cudnn-windows-x86_64-8.9.6.25_cuda11-archive.zip'.\");");
        println!("console.log(\"4. Extract the contents of the downloaded zip file.\");");
        println!("console.log(\"5. Copy the following files to the CUDA installation directory:\");");
        println!("console.log(\"   - Copy 'cudnn64_8.dll' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\bin'\");");
        println!("console.log(\"   - Copy 'cudnn.h' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\include'\");");
        println!("console.log(\"   - Copy 'cudnn.lib' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\lib\\\\x64'\");");
        println!("console.log(\"6. Press any key to continue once you have moved and unzipped the folder.\");");
        println!("console.log(\"Note: If you encounter issues, you might need to manually add the following paths to your system PATH:\");");
        println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\bin'\");");
        println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\libnvvp'\");");
        println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\lib'\");");
        
        // Config.py
        println!("Config.py:");
        println!("# Configuration for BetterCam Screen Capture and YOLO model");
        println!("# Screen Capture Settings");
        println!("screenWidth = 480  # Updated screen width for better resolution");
        println!("screenHeight = 480  # Updated screen height for better resolution");
        println!("# Object Detection Settings");
        println!("confidenceThreshold = 0.5  # Confidence threshold for object detection");
        println!("nmsThreshold = 0.4  # Non-max suppression threshold to filter overlapping boxes");
        println!("# YOLO Model Settings");
        println!("modelType = 'onnx'  # Choose 'torch' or 'onnx' based on the model you want to load");
        println!("torchModelPath = 'models/fn_v5.pt'  # Path to YOLOv5 PyTorch model");
        println!("onnxModelPath = 'models/fn_v5.onnx'  # Path to YOLOv5 ONNX model");
        println!("# BetterCam Settings");
        println!("targetFPS = 60  # Frames per second for capturing");
        println!("maxBufferLen = 512  # Max buffer length for storing frames");
        println!("region = None  # Region for capture (set to None for full screen)");
        println!("useNvidiaGPU = True  # Set to True to enable GPU acceleration if available");
        println!("# Colors for Bounding Boxes");
        println!("boundingBoxColor = (0, 255, 0)  # Default bounding box color in BGR format");
        println!("highlightColor = (0, 0, 255)  # Color for highlighted objects\n");


        // Python Code Explanation
        println!("Python Code Explanation:");
        println!("The following Python code snippet provides instructions for installing the required Python dependencies:");
        println!("The Python code has been written in Python 3.11.6\n");

        // Contributing
        println!("Contributing:");
        println!("Contributions are welcome! Please fork the repository and submit a pull request with your changes.\n");

        // License
        println!("License:");
        println!("This project is proprietary and all rights are reserved by the author.");
        println!("Unauthorized copying, distribution, or modification of this project is strictly prohibited./n"); 
        println!("Unless You have written permission from the Developer or the FNBUBBLES420 ORG.\n");

        // Developers
        println!("Developers:");
        println!("1. Bubbles The Dev\n");

        // Members
        println!("Members:");
        println!("1. Bubbles The Dev");
        println!("2. phillipealaksa\n");

        // Triage
        println!("Triage:");
        println!("1. phillipealaksa\n");

        // Organization
        println!("Organization:");
        println!("1. FNBUBBLES420 ORG\n");

        // Contact
        println!("Contact:");
        println!("For any questions or issues, please open an issue on the GitHub repository or contact the maintainers directly.\n");

        // Additional line
        println!("Welcome to Game Vision Aid!");
    }
