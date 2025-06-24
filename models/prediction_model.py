import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation
from torch.nn import Linear




class MLPNet(torch.nn.Module):
    def __init__(self, 
                 input_dims, output_dim,
                 hidden_layer_sizes=(64,),
                 hidden_activation='relu',
                 output_activation=None,
                dropout=0.0):
        super(MLPNet, self).__init__()

        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)
         
           ##创建隐藏层
        for layer_size in hidden_layer_sizes:
              
            hidden_dim = layer_size
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                get_activation(hidden_activation),  ##获取激活函数
                nn.Dropout(dropout),
            )
            layers.append(layer)
            input_dim = hidden_dim

        ##创建最后的输出层
        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            get_activation(output_activation) if output_activation else nn.Identity(),
        )
        layers.append(layer)
        
        self.layers = layers

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]   ##如果输入是一个张量，转为列表
        input_var = torch.cat(inputs, -1)    # 将所有输入拼接起来
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var

class MLPNet2(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(MLPNet2, self).__init__()
        self.lin1 = Linear(in_feats,hidden_size)
        self.lin2 = Linear(hidden_size,hidden_size//2)
        self.lin3 = Linear(hidden_size//2,hidden_size//4)
        # self.lin4=  Linear(hidden_size//4,hidden_size//8)
        self.lin5=  Linear(hidden_size//4,1)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin5(x))
        return x
