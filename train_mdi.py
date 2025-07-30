import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd
import json

from training.gnn_mdi import train_gnn_mdi
from utils.utils import auto_select_gpu
from custom.custom_data import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_path', type=str, default='missing_data.csv')
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
    parser.add_argument('--epochs', type=int, default=30)
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
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--save_prediction', action='store_true', default=True)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    args = parser.parse_args()

    print("config:", args)

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

    imputed_values, model, impute_model = train_gnn_mdi(data, args, log_path, device)

    # === 保存模型参数和结构配置 ===
    torch.save(model.state_dict(), osp.join(log_path, 'model_1.pth'))
    torch.save(impute_model.state_dict(), osp.join(log_path, 'model_2.pth'))

    model_config = {
        'model_types': args.model_types,
        'node_dim': args.node_dim,
        'edge_dim': args.edge_dim,
        'edge_mode': args.edge_mode,
        'impute_hiddens': args.impute_hiddens,
        'impute_activation': args.impute_activation,
        'concat_states': args.concat_states,
        'dropout': args.dropout
    }
    with open(osp.join(log_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f)

    # === 进行缺失值填补 ===
    output_matrix = data.full_matrix.clone()
    unknown_idx = np.where(np.isnan(output_matrix.cpu().numpy()))
    for (i, j), val in zip(zip(*unknown_idx), imputed_values):
        output_matrix[i, j] = val.item()

    df_filled = pd.DataFrame(output_matrix.cpu().numpy())
    df_filled.to_csv("imputed_result.csv", index=False)
    print("output: imputed_result.csv")

if __name__ == '__main__':
    main()
