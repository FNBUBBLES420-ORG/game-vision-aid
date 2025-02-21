# engine

python .\export.py --weights ./"your_model_path.pt" --include engine --half --imgsz 320 320 --device 0


# onnx 

python .\export.py --weights ./"your_model_path.pt" --include onnx --half --imgsz 320 320 --device 0
