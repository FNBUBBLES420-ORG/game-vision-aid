# Configuration for BetterCam Screen Capture and YOLO model

# Screen Capture Settings
screenWidth = 480  # Updated screen width for better resolution - (recommended 480)
screenHeight = 480  # Updated screen height for better resolution - (recommended 480)

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
