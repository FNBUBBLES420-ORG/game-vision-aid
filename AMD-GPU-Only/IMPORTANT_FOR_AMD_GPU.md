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


### Troubleshooting

If you encounter any issues, such as an **Ultralytics error**, follow the steps below:

1. Run the following command in `CMD.exe` to upgrade Ultralytics:
    ```
    pip install --upgrade ultralytics
    ```

2. The `Ultralytics` package is already included in the `requirements.txt` and `requirements.bat` files.

3. Use the `update_ultralytics.bat` script if you continue to experience Ultralytics errors.

### Note 
  - If you get an Ultralytics error when installing 
  Run the Command below in `CMD.exe`
    ```
    pip install --upgrade ultralytics
    ```
  - I did include the `Ultralytics` in the requirements.txt` and `requirements.bat`.
  - Use the `install_pytorch.bat` as it is tied to `cu118`. (Recommended)
  - Use the `update_ultralytics.bat` if you have `ultralytics error`



## If you have an AMD GPU and want to use ROCm:
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.0
```
