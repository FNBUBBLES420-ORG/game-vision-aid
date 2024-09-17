# Exporting For AMD GPU Users


## 💻 For `AMD GPU` Users

If you have an **AMD GPU**, follow the steps below to set up DirectML support for faster processing:

1. **Install AMD GPU dependencies**:
   - Run the **`amd_gpu_requirements.bat`** script to install the necessary dependencies for AMD GPU:
     ```
     amd_gpu_requirements.bat
     ```

   This will install **DirectML** and other required packages, enabling the project to run efficiently on AMD hardware.

   
2. `CD` the `export.pt` in `CMD.exe`
3. If you want to `CD` the `export.py` in the `Export-Models` folder you can so you can keep track of your models
4. 
- type the following command:
```
python .\export.py --weights ./"your_model_path.pt" --include onnx --half --imgsz 320 320 --device 0
```
























