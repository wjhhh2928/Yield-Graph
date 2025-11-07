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

---

## 🛠️ Installation

### 1. Environment Setup (Python 3.8, CUDA 11.8)

This project assumes:

- Python **3.8**
- PyTorch with **CUDA 11.8** support

A typical setup with `conda`:

```bash
# (1) Create and activate a virtual environment
conda create -n gnn-maize python=3.8
conda activate gnn-maize

# (2) Install PyTorch (CUDA 11.8)
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# (3) Install graph learning packages
pip install torch-geometric==2.6.1 torch-scatter==2.1.2 torch-sparse==0.6.18

# (4) Install core data & ML libraries
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl optuna xgboost lightgbm \
            shapely geopandas fiona pyproj

# (5) (Optional) Visualization & web tools
pip install pyecharts cairosvg playwright selenium

# (6) (Optional) Table & logging utilities
pip install prettytable colorlog

# (7) Quick sanity check
python -c "import torch; print(torch.cuda.is_available())"
# You should see:
# True

# (8) System notes
# This environment assumes CUDA 11.8 and an NVIDIA driver version ≥ 520.x.
# On Ubuntu, you may also need:
#   sudo apt install python3.8-dev libgl1-mesa-glx libglib2.0-0
#
# Alternatively, if a requirements.txt is provided:
#   pip install -r requirements.txt

---
fff

