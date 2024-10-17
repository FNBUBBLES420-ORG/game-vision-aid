# Configuration for BetterCam Screen Capture and YOLO model

# Screen Capture Settings
screenWidth = 480  # Updated screen width for better resolution - (recommended 480)
screenHeight = 480  # Updated screen height for better resolution - (recommended 480)

# Object Detection Settings
confidenceThreshold = 0.5  # Confidence threshold for object detection
nmsThreshold = 0.4  # Non-max suppression threshold to filter overlapping boxes

# YOLO Model Selection
# Choose the type of model you want to use: 'torch', 'onnx', or 'engine'
# 'torch' is for PyTorch models (.pt), 'onnx' is for ONNX models (.onnx), and 'engine' is for TensorRT models (.engine)
modelType = 'onnx'  # Example: Set to 'torch' for PyTorch models, 'onnx' for ONNX models, 'engine' for TensorRT models

# Model Paths (YOLOv5 and YOLOv8)
# Uncomment the model path corresponding to the version of YOLO you want to use
# YOLOv5 PyTorch and ONNX models
torchModelPath = 'models/fn_v5.pt'  # YOLOv5 PyTorch model path (.pt)
onnxModelPath = 'models/fn_v5.onnx'  # YOLOv5 ONNX model path (.onnx)

# YOLOv8 PyTorch and ONNX models
# torchModelPath = 'models/fn_v8.pt'  # YOLOv8 PyTorch model path (.pt)
# onnxModelPath = 'models/fn_v8.onnx'  # YOLOv8 ONNX model path (.onnx)

# TensorRT Model
# tensorrtModelPath = 'models/fn_model.engine'  # Path to TensorRT model (.engine)

# BetterCam Settings
targetFPS = 60  # Frames per second for capturing
maxBufferLen = 512  # Max buffer length for storing frames
region = None  # Region for capture (set to None for full screen)
useNvidiaGPU = True  # Set to True to enable GPU acceleration if available
monitorIdx = 0  # Index for multi-monitor support, set 0 for primary monitor

# Colors for Bounding Boxes
boundingBoxColor = (0, 255, 0)  # Default bounding box color in BGR format
highlightColor = (0, 0, 255)  # Color for highlighted objects

# Overlay Settings
overlayWidth = 1920  # Overlay window width
overlayHeight = 1080  # Overlay window height
overlayAlpha = 0.6  # Transparency of overlay

# GPU Support
useCuda = True  # Enable CUDA support if available
useDirectML = False  # Set to True to enable DirectML for AMD GPUs
