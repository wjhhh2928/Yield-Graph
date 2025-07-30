import argparse
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from custom.custom_data import load_data
from utils.utils import auto_select_gpu

@torch.no_grad()
def run_inference(args, device):
    print("\U0001F50D Loading data from:", args.custom_path)
    data = load_data(args, custom_path=args.custom_path)
    log_path = f"./custom/test/{args.log_dir}/"

    # === Load model structure ===
    print("\U0001F4E6 Loading trained models...")
    model = get_gnn(data, args).to(device)

    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))

    input_dim = args.node_dim * len(model.convs) * 2 if args.concat_states else args.node_dim * 2
    output_dim = 1

    impute_model = MLPNet(
        input_dim,
        output_dim,
        hidden_layer_sizes=impute_hiddens,
        hidden_activation=args.impute_activation,
        dropout=args.dropout
    ).to(device)

    # Load weight parameters (not full model object)
    model.load_state_dict(torch.load('model_1.pt', map_location=device), strict=False)
    impute_model.load_state_dict(torch.load('model_2.pt', map_location=device), strict=False)

    model.eval()
    impute_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    target_edge_index = data.edge_index.to(device)  #predict

    print("\U0001F680 Running inference...")
    x_embd = model(x, edge_attr, edge_index)
    pred = impute_model([x_embd[target_edge_index[0]], x_embd[target_edge_index[1]]])
    pred_values = pred[:int(target_edge_index.shape[1] / 2), 0].cpu().numpy()

    print("Filling missing values and saving result...")
    output_matrix = data.full_matrix.clone()
    output_matrix_np = output_matrix.cpu().numpy()
    unknown_idx = np.where(np.isnan(output_matrix_np))

    if len(pred_values) < len(unknown_idx[0]):
        print("Warning: Number of predictions is less than number of missing values, unable to fully impute!")

    for (i, j), val in zip(zip(*unknown_idx), pred_values):
        output_matrix_np[i, j] = val

    df_filled = pd.DataFrame(output_matrix_np)
    df_filled.to_csv(args.output_csv, index=False)
    print(f"Done! Saved to: {args.output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_path', type=str, default='miss_data.csv', help='Path to CSV file with missing values')
    parser.add_argument('--log_dir', type=str, default='custom0', help='Directory containing saved models')
    parser.add_argument('--output_csv', type=str, default='imputed.csv', help='Output filled CSV file')

    #  Model parameters (must match training)
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
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--known', type=float, default=1.0, help='Proportion of known edges used')

    args = parser.parse_args()

    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        device = torch.device(f"cuda:{cuda}")
        print(f"Using GPU: {cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    run_inference(args, device)


if __name__ == '__main__':
    main()
