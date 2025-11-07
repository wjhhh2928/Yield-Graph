# 🌽 Yield-Graph: Multi-stage Growth-aware Maize Yield Prediction via Graph Neural Networks

> **⚠️ Important notice (re Mul-PheG2P link)**  
> This repository was mistakenly linked in the supplementary materials of the manuscript  
> **“Mul-PheG2P: Decoupled learning and prediction-space fusion enables robust and interpretable multi-phenotype genomic prediction.”**  
> Please note that **this project is not associated with that manuscript**.  
>
> For the correct code and data corresponding to the Mul-PheG2P paper, please visit the official repository:  
> 👉 [https://github.com/wjhhh2928/Mul-PheG2P](https://github.com/wjhhh2928/Mul-PheG2P)  
>
> We apologize for any confusion caused and appreciate your understanding.

---

## 📘 Overview

This repository contains the code for:

> **Yield-Graph: Multi-stage Growth-aware Maize Yield Prediction via Graph Neural Networks**  
> by *Jiahui Wang, Yong Zhang, Yuqing Zhang, Xinglin Piao, Aiwen Wang, Xiangyu Zhao, Kaiyi Wang*.

The project focuses on:

- Building **graph-based representations** of maize traits and environments across growth stages.  
- Performing **trait imputation** via graph neural networks to recover missing phenotypic values.  
- Using **imputed traits + environmental features** to predict maize yield.

<p align="center">
    <img src="Fig1.png" alt="Model" width="85%">
</p>
---

## 🛠️ Installation

###  Environment Setup (Python 3.8, CUDA 11.8)

This project assumes:

- Python **3.8**
- PyTorch with **CUDA 11.8** support

A typical setup with `conda`:


(1) Create and activate a virtual environment

```bash
conda create -n gnn-maize python=3.8
conda activate gnn-maize
```

(2) Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```
(3) Install graph learning packages
```bash
pip install torch-geometric==2.6.1 torch-scatter==2.1.2 torch-sparse==0.6.18
```
(4) Install core data & ML libraries
```bash
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl optuna xgboost lightgbm \
            shapely geopandas fiona pyproj
```
(5) (Optional) Visualization & web tools
```bash
pip install pyecharts cairosvg playwright selenium
```
(6) (Optional) Table & logging utilities
```bash
pip install prettytable colorlog
```
(7) Quick sanity check
```bash
python -c "import torch; print(torch.cuda.is_available())"
# You should see:
# True
```

(8) System notes
This environment assumes CUDA 11.8 and an NVIDIA driver version ≥ 520.x.
On Ubuntu, you may also need:
```bash
sudo apt install python3.8-dev libgl1-mesa-glx libglib2.0-0
```
Alternatively, if a requirements.txt is provided:
```bash
pip install -r requirements.txt
```

## ▶️ Running the Code 

This project includes three main scripts:
```bash
train_mdi.py – Train trait imputation model

predict.py – Trait imputation (fill missing phenotypes)

train_test.py – Maize yield prediction
```
Please ensure all required dependencies are installed before running.
(1) Train Trait Imputation Model (train_mdi.py) This script trains a graph-based model to impute missing phenotypic traits using: Graph structure (e.g., bipartite or multi-relational graphs) Feature relationships across traits and stages Example:
```bash
python train_mdi.py
```
(2) Trait Imputation (predict.py) This script uses the trained imputation model to fill in missing trait values. Input: raw or partially observed phenotypic data Output: a completed trait matrix with imputed values Example:
```bash
python predict.py
```
 (3) Maize Yield Prediction (train_test.py) This script trains and evaluates the yield prediction model using: Imputed traits from the previous step Environmental variables (e.g., weather, management) Possibly multi-stage phenotyping data Example:
```bash
python train_test.py
```
The script will: Train the model on available data Evaluate prediction performance on a held-out test set Optionally export metrics and prediction results (depending on your implementation)

## 📂 Project Structure (Typical) A possible structure (for orientation) might look like:
```bash
      .
      ├── train_mdi.py          # Trait imputation training
      ├── predict.py            # Trait imputation / missing value filling
      ├── train_test.py         # Maize yield prediction
      ├── data/                 # Input data (not included in this repo)
      ├── models/               # Model definitions (if applicable)
      ├── utils/                # Utility functions, preprocessing, etc.
      └── requirements.txt      # Optional: full dependency list
```
Please refer to the inline comments and function docstrings in each script for dataset formats and additional configuration details.

## 📌 Notes
Make sure your data paths and file formats match what the scripts expect (e.g. CSV/Excel for traits & environments, possibly NPZ/CSV for graphs).
If you encounter version issues with torch-geometric / torch-scatter / torch-sparse,
please consult the official PyG installation guide and align versions with your PyTorch & CUDA stack. 
Large-scale graph training can be GPU- and memory-intensive; adjust batch sizes and model sizes accordingly.

## 📜 Citation
If you use this code in your research, please cite the corresponding maize yield prediction work by:
```bash
Jiahui Wang, Yong Zhang, Yuqing Zhang, Xinglin Piao, Aiwen Wang, Xiangyu Zhao, Kaiyi Wang
Yield-Graph: Multi-stage Growth-aware Maize Yield Prediction via Graph Neural Networks.
(Full bibliographic details to be added when available.)
```
