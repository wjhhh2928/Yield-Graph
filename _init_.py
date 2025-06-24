# /home/zcy/zyq/grape_y/__init__.py
from .models.gnn_model import get_gnn
from .models.prediction_model import MLPNet, MLPNet2

__all__ = ["get_gnn", "MLPNet", "MLPNet2"]
