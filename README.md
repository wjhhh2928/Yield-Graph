This is the code for Graph neural network-based prediction of maize yield across multiple growth stages by Jiahui Wang, Yong Zhang, Yuqing Zhang, Xinglin Piao, Aiwen Wang, Xiangyu Zhao, Kaiyi Wang.

# Installation
Getting Started: Environment Setup (Python 3.8, CUDA 11.8)
This project requires Python 3.8 and PyTorch with CUDA 11.8 support. A typical setup might look like:
# (1) Create and activate a virtual environment (conda recommended)
conda create -n name python=3.8
conda activate name

# (2) Install core PyTorch components with CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# (3) Install required machine learning and graph packages
pip install torch-geometric==2.6.1 torch-scatter==2.1.2 torch-sparse==0.6.18

# (4) Install supporting libraries for data processing and analysis
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl optuna xgboost lightgbm shapely geopandas fiona pyproj

# (5) Visualization and web scraping tools (optional)
pip install pyecharts cairosvg playwright selenium

# (6) Optional table and logging support
pip install prettytable colorlog

# (7) Final checks
python -c "import torch; print(torch.cuda.is_available())"
Notes:
This environment uses CUDA 11.8, compatible with NVIDIA driver ≥ 520.x.

Some packages like torch-geometric require specific torch-scatter / torch-sparse versions.

If you're on Ubuntu, make sure to have the following system libraries installed:

bash
sudo apt install python3.8-dev libgl1-mesa-glx libglib2.0-0

# Running the Code
This project includes two main scripts for trait imputation and maize yield prediction:

train_mdi.py – Train Trait Imputation Model

predict.py – Trait Imputation
Runs the model to fill in missing phenotypic values using bipartite graph and feature-based inference.

train_test.py – Yield Prediction
Trains and evaluates the model for maize yield prediction based on imputed traits and environmental data.

Example usage:

# Step 1: Train trait imputation model
python train_mdi.py

# Step 2: Fill missing trait values
python predict.py

# Step 2: Predict maize yield
python Train_test.py
Make sure all dependencies are installed from requirements.txt before running.
