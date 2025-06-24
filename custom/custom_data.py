import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_data(args):
    csv_path = args.custom_path
    df = pd.read_csv(csv_path)
    data_matrix = df.values.astype(np.float32)

    num_samples, num_features = data_matrix.shape

   
    observed_mask = ~np.isnan(data_matrix)
    observed_indices = np.array(np.where(observed_mask)).T  # (n_edges, 2)
    values = data_matrix[observed_mask]  # (n_edges,)

  
    row = observed_indices[:, 0]
    col = observed_indices[:, 1] + num_samples
    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(values, dtype=torch.float)

    
    num_edges = edge_index.size(1)
    perm = np.random.permutation(num_edges)
    train_size = int(args.known * num_edges)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    train_edge_index = edge_index[:, train_idx]
    train_edge_attr = edge_attr[train_idx]
    train_labels = edge_attr[train_idx]

    test_edge_index = edge_index[:, test_idx]
    test_edge_attr = edge_attr[test_idx]
    test_labels = edge_attr[test_idx]

    
    x = torch.ones((num_samples + num_features, 1), dtype=torch.float)

    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        train_labels=train_labels,
        test_edge_index=test_edge_index,
        test_edge_attr=test_edge_attr,
        test_labels=test_labels,
        num_nodes=num_samples + num_features,
        num_samples=num_samples,
        num_features=num_features,
        full_matrix=torch.tensor(data_matrix, dtype=torch.float)
    )

    
    data.num_node_features = x.shape[1]
    data.edge_attr_dim = 1

    return data
