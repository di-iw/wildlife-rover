環境構築  
powershellで使えるように
- conda init powershell

環境作成
- conda create -n wildlife-rover -c conda-forge python=3.11 pybullet=3.25 "numpy<2" pip -y
- conda activate wildlife-rover

- python -m pip install --upgrade pip
- python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
- python -m pip install ultralytics opencv-python

確認
- python -c "import sys; print('Python:', sys.version)"
    →3.11.18
- python -c "import pybullet as p; import pybullet_data; print('PyBullet:', p.getAPIVersion()); print('Data:', pybullet_data.getDataPath())"
    →pybullet build time: Oct 21 2025 11:46:21, PyBullet: 202010061
- python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    →Torch: 2.15.0+cpu, CUDA available: False
- python -c "import cv2; print('OpenCV:', cv2.__version__)"
    →OpenCV: 5.0.0
- python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
    →Ultralytics: 8.4.194

HuskyのURDF確認
- python -c "from pathlib import Path; import pybullet_data; p=Path(pybullet_data.getDataPath())/'husky'/'husky.urdf'; print(p, p.exists())"
    →True

conda環境書き出し
- conda env export > environment.yml