# YOLO Export Script for AMD GPU Users

This guide will walk you through converting a PyTorch (`.pt`) model to ONNX format using the [YOLO Export Script](https://github.com/KernFerm/yolo-export-script), specifically tailored for AMD GPU users.

## Prerequisites

Before you begin, ensure that you have the following installed:

- **PyTorch**: With ROCm support for AMD GPUs
- **ONNX**: The Open Neural Network Exchange format
- **ROCm**: AMD's GPU Open Compute platform, if you're using PyTorch with ROCm support
- **Python 3.8+**

## Step-by-Step Guide for Converting `.pt` to `.onnx`

### 1. Clone the YOLO Export Script

Start by cloning the [YOLO Export Script repository](https://github.com/KernFerm/yolo-export-script) to your local machine:

```
git clone https://github.com/KernFerm/yolo-export-script
cd yolo-export-script
```

## 2. Set Up Your Environment

Ensure your Python environment is set up with the required packages. You can install them by running:

```
pip install -r requirements.txt
```

## The Key Dependencies for This Script Are:

- **torch**: Required for handling PyTorch models.
- **onnx**: For exporting models to ONNX format.
- **numpy**: For numerical operations and array manipulation.

## 3. Convert `.pt` to `.onnx`

The script in the repository allows you to export a YOLOv5 model (trained or untrained) from PyTorch format (`.pt`) to ONNX format. This is useful for deployment on frameworks that support ONNX but not PyTorch.

To convert a model, run the following command:

```
python export.py --weights your_model.pt --img-size 640 --batch-size 1 --device cpu --simplify
```


























