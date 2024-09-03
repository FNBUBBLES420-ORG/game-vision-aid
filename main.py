import sys
import numpy as np
import cv2  # OpenCV for image processing and displaying the overlay
import torch  # PyTorch for running YOLOv5 models
import onnxruntime as ort  # ONNX Runtime for running ONNX models
import time
import mss  # MSS for fast screen capturing

def load_model(model_path='ultralytics/yolov5s'):
    """
    Load the YOLOv5 model.
    :param model_path: Path to the custom model (.pt or .onnx) or model name from the YOLOv5 repository.
    :return: Loaded YOLOv5 model or ONNX session.
    """
    try:
        start_time = time.time()  # Start timing
        if model_path.endswith('.pt'):
            print("Loading PyTorch model...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # Custom model
            model_type = 'torch'    # YOLOv5 model
        elif model_path.endswith('.onnx'):  # ONNX model
            print("Loading ONNX model...")
            model = ort.InferenceSession(model_path)    # ONNX model
            model_type = 'onnx'
        else:
            print("Loading YOLOv5 model from repository...")
            model = torch.hub.load('ultralytics/yolov5', model_path, force_reload=True) # YOLOv5 model from the repository
            model_type = 'torch'    # YOLOv5 model
        end_time = time.time()  # End timing
        print(f"Model loaded in {end_time - start_time:.2f} seconds")  # Print the loading time
        print("Model loaded and ready.")
        print("You can now use the script.")
        print("if you have any issues using this join the discord https://discord.fnbubbles420.org/invite HEAD TO 🧑🏫-teaching_tutoring")
        return model, model_type    # Return the loaded model and model type
    except Exception as e:  # Catch any exceptions during model loading
        print(f"Error loading model: {e}")  # Print the error message
        sys.exit(1)

def capture_screen():
    """
    Captures the screen using MSS.
    :return: Captured screen frame as a numpy array.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor
        img = sct.grab(monitor)
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

def detect_objects(model, model_type, frame, device):
    """
    Detects objects in the frame using YOLOv5 or ONNX model.
    :param model: YOLOv5 or ONNX model.
    :param model_type: Type of the model ('torch' or 'onnx').
    :param frame: The frame captured from the screen.
    :param device: Device to run the model on (CPU or GPU).
    :return: Results of detection.
    """
    try:
        if model_type == 'torch':
            frame_tensor = torch.from_numpy(frame).to(device)
            results = model(frame_tensor)
        elif model_type == 'onnx':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 640))
            input_tensor = frame_resized.astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0).transpose(0, 3, 1, 2)
            input_tensor /= 255.0
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
            results = outputs[0]
        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def draw_bounding_boxes(frame, results, color, model_type):
    """
    Draws bounding boxes on the frame.
    :param frame: The frame to draw on.
    :param results: Detection results from YOLOv5 or ONNX model.
    :param color: Color of the bounding boxes.
    :param model_type: Type of the model ('torch' or 'onnx').
    :return: Frame with bounding boxes drawn.
    """
    if results is not None:
        if model_type == 'torch':
            for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results.xyxy[0]):
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        elif model_type == 'onnx':
            for i in range(len(results)):
                if len(results[i]) >= 5:  # Ensure there are at least 5 values to unpack
                    xmin, ymin, xmax, ymax, conf = results[i][:5]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                else:
                    print(f"Skipping result {i} due to insufficient values: {results[i]}")
    return frame

def get_color_from_input():
    """
    Function to get a color from user input.
    :return: Selected color in BGR format.
    """
    color_dict = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
        "brown": (42, 42, 165),
        "gray": (128, 128, 128)
    }

    while True:
        color_input = input("Enter the overlay color (e.g., red, green, blue) or type 'exit' or 'q' to quit: ").strip().lower()
        if color_input in color_dict:
            return color_dict[color_input]
        elif color_input in ["exit", "q"]:
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid color name. Please try again.")

def main():
    # Ensure the game is running and visible on the screen before starting the script
    input("Make sure the game is running and visible on the screen. Press Enter to continue...")

    # Check for NVIDIA GPU availability with CUDA support
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA-enabled NVIDIA GPU found. Using GPU acceleration.")
    else:
        device = 'cpu'
        print("No CUDA-enabled NVIDIA GPU found. Using CPU.")

    # Load YOLOv5 or ONNX model
    model_path = 'models/custom_model.onnx'  # Change to your custom model path if needed
    start_time = time.time()  # Start timing for model loading
    model, model_type = load_model(model_path)
    if model_type == 'torch':
        model = model.to(device)
    end_time = time.time()  # End timing for model loading
    print(f"Model loading time: {end_time - start_time:.2f} seconds")

    # Get a color for the overlay from user input
    print("Getting overlay color from user input...")
    start_time = time.time()  # Start timing for color input
    overlay_color = get_color_from_input()
    end_time = time.time()  # End timing for color input
    print(f"Color input time: {end_time - start_time:.2f} seconds")
    print("Overlay color selected.")

    # Main loop
    try:
        while True:
            # Capture the screen
            frame = capture_screen()
            print("Screen captured.")

            # Detect objects in the frame
            results = detect_objects(model, model_type, frame, device)
            print("Objects detected.")

            # Draw bounding boxes around detected objects with the chosen color
            frame_with_boxes = draw_bounding_boxes(frame, results, overlay_color, model_type)
            print("Bounding boxes drawn.")

            # Save the frame with the overlay to disk (optional)
            # cv2.imwrite("output_frame.jpg", frame_with_boxes)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
