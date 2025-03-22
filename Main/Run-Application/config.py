# Configuration for BetterCam Screen Capture and YOLO model

# 🔍 Screen Capture Settings
screenWidth = 640     # Width of the screen region to capture
screenHeight = 640    # Height of the screen region to capture

# 🎯 Object Detection Settings
confidenceThreshold = 0.5  # Minimum confidence to draw detection
nmsThreshold = 0.4         # Non-max suppression threshold

# 📦 YOLO Model Selection
# Options: 'torch' (YOLOv5/v8 .pt), 'onnx' (.onnx), or 'engine' (.engine for TensorRT)
modelType = 'torch'

# 🧠 Model File Paths
torchModelPath = 'models/yolov8n.pt'           # PyTorch model (YOLOv5/YOLOv8)
onnxModelPath = 'models/yolov8n.onnx'          # ONNX model path
tensorrtModelPath = 'models/yolov8n.engine'    # TensorRT engine file

# 🎥 BetterCam Capture Settings
targetFPS = 60            # Target frames per second
maxBufferLen = 512        # Max buffer size for frame storage
region = None             # Set to None for full-screen capture
monitorIdx = 0            # Use 0 for primary monitor

# 🖍️ Drawing Bounding Boxes
boundingBoxColor = (0, 255, 0)  # BGR Green
highlightColor = (0, 0, 255)    # BGR Red for highlights

# 🪟 Overlay Settings
overlayWidth = 1920
overlayHeight = 1080
overlayAlpha = 200  # 0 (fully transparent) to 255 (fully opaque)

# ⚡ GPU Backend Preferences
useCuda = True       # Enable CUDA (for NVIDIA GPUs)
useDirectML = True   # Enable DirectML (for AMD GPUs or fallback)
