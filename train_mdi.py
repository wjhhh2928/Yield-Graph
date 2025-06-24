import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_mdi import train_gnn_mdi
from utils.utils import auto_select_gpu
from custom.custom_data import load_data  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_path', type=str, default='missing_data.csv', help='Path to custom CSV data') 
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None)
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None)
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default=0)
    parser.add_argument('--valid', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='custom0')
    parser.add_argument('--mode', type=str, default='train', help='运行模式，支持 train/debug')
    parser.add_argument('--save_prediction', action='store_true', default=True)
    parser.add_argument('--transfer_dir', type=str, default=None)  
    parser.add_argument('--transfer_extra', type=str, default='')  
    args = parser.parse_args()

    print("config：", args)

    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args)
    log_path = './custom/test/{}/'.format(args.log_dir)
    os.makedirs(log_path, exist_ok=True)

    imputed_values = train_gnn_mdi(data, args, log_path, device)

    output_matrix = data.full_matrix.clone()
    edge_idx = data.edge_index.cpu().numpy()
    unknown_idx = np.where(torch.isnan(output_matrix.cpu().numpy()))

    for (i, j), val in zip(zip(*unknown_idx), imputed_values):
        output_matrix[i, j] = val.item()

    df_filled = pd.DataFrame(output_matrix.cpu().numpy())
    df_filled.to_csv("imputed_result.csv", index=False)
    print("out_put: imputed_result.csv")

if __name__ == '__main__':
    main()