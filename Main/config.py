# Configuration for BetterCam Screen Capture and YOLO model

# Screen Capture Settings
screenWidth = 640
screenHeight = 640

# Object Detection Settings
confidenceThreshold = 0.5
nmsThreshold = 0.4

# YOLO Model Selection
# Choose one: 'torch', 'onnx', or 'engine'
modelType = 'torch'

# Model Paths
torchModelPath = 'models/yolov8n.pt'           # Supports YOLOv5/YOLOv8 PyTorch models
onnxModelPath = 'models/yolov8n.onnx'           # ONNX model
tensorrtModelPath = 'models/yolov8n.engine'     # TensorRT engine

# BetterCam Settings
targetFPS = 60
maxBufferLen = 512
region = None
useNvidiaGPU = True
monitorIdx = 0

# Colors for Bounding Boxes
boundingBoxColor = (0, 255, 0)
highlightColor = (0, 0, 255)

# Overlay Settings
overlayWidth = 1920
overlayHeight = 1080
overlayAlpha = 200  # Range: 0–255

# GPU Support
useCuda = True
useDirectML = True
