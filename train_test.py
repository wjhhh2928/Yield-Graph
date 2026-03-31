import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.FloatTensor)
from torch.cuda.amp import autocast, GradScaler 
from torch.autograd import Variable
from pandas.core.frame import DataFrame
from networkx import karate_club_graph,to_numpy_array  
import matplotlib.pyplot as plt
import networkx as nx
import torch as t
import torch.nn.functional as F
import scipy.sparse as sp

from torch.nn import Linear

import csv 
import scipy.sparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_dense_adj
from collections import Counter
from torch.nn.parameter import Parameter
import math 
import os
from sklearn.model_selection import KFold 

from sklearn.preprocessing import StandardScaler 


corn_file = open('data.csv') 
csv_reader_lines = csv.reader(corn_file)

data_select_all = []
data_graph_all = []
data_target_all = []

num = 0

for row in csv_reader_lines: 
    
    if num == 0:
        num = num + 1
        continue

    num = num + 1
    
    
    try:
        muchan = float(row[-1])
        
        
        feature_1 = float(row[3])
        feature_2 = float(row[4])
        feature_3 = float(row[5])
        feature_4 = float(row[6])
        feature_5 = float(row[7])
        feature_6 = float(row[8])
        feature_7 = float(row[9])
        feature_8 = float(row[10])
        feature_9 = float(row[11])
        feature_10 = float(row[12])
        feature_11 = float(row[13])
        feature_12 = float(row[14])
        feature_13 = float(row[15])
        feature_14 = float(row[16])
        feature_15 = float(row[17])
        feature_16 = float(row[18])
        feature_17 = float(row[19])
        feature_18 = float(row[20])
        feature_19 = float(row[21])
        feature_20 = float(row[22])
        feature_21 = float(row[23])
        feature_22 = float(row[24])
        feature_23 = float(row[25])
        feature_24 = float(row[26])
        feature_25 = float(row[27]) 
        feature_26 = float(row[28])
        feature_27 = float(row[29])
        feature_28 = float(row[30])
        feature_29 = float(row[31])
        feature_30 = float(row[32])
        feature_31 = float(row[33])
        feature_32 = float(row[34])
        feature_33 = float(row[35])
        feature_34 = float(row[36])
        feature_35 = float(row[37]) 
        feature_36 = float(row[38])
    except IndexError:
        print(f"Skipping row {num} due to insufficient columns.")
        continue
    except ValueError:
        print(f"Skipping row {num} due to value error.")
        continue


    # log 
    feature_1 = np.log(feature_1+1) if feature_1>=0 else -np.log(abs(feature_1)+1)
    feature_2 = np.log(feature_2+1) if feature_2>=0 else -np.log(abs(feature_2)+1)
    feature_3 = np.log(feature_3+1) if feature_3>=0 else -np.log(abs(feature_3)+1)
    feature_4 = np.log(feature_4+1) if feature_4>=0 else -np.log(abs(feature_4)+1)
    feature_5 = np.log(feature_5+1) if feature_5>=0 else -np.log(abs(feature_5)+1)
    feature_6 = np.log(feature_6+1) if feature_6>=0 else -np.log(abs(feature_6)+1)
    feature_7 = np.log(feature_7+1) if feature_7>=0 else -np.log(abs(feature_7)+1)
    feature_8 = np.log(feature_8+1) if feature_8>=0 else -np.log(abs(feature_8)+1)
    feature_9 = np.log(feature_9+1) if feature_9>=0 else -np.log(abs(feature_9)+1)
    feature_10 = np.log(feature_10+1) if feature_10>=0 else -np.log(abs(feature_10)+1)
    feature_11 = np.log(feature_11+1) if feature_11>=0 else -np.log(abs(feature_11)+1)
    feature_12 = np.log(feature_12+1) if feature_12>=0 else -np.log(abs(feature_12)+1)
    feature_13 = np.log(feature_13+1) if feature_13>=0 else -np.log(abs(feature_13)+1)
    feature_14 = np.log(feature_14+1) if feature_14>=0 else -np.log(abs(feature_14)+1)
    feature_15 = np.log(feature_15+1) if feature_15>=0 else -np.log(abs(feature_15)+1)
    feature_16 = np.log(feature_16+1) if feature_16>=0 else -np.log(abs(feature_16)+1)
    feature_17 = np.log(feature_17+1) if feature_17>=0 else -np.log(abs(feature_17)+1)
    feature_18 = np.log(feature_18+1) if feature_18>=0 else -np.log(abs(feature_18)+1)
    feature_19 = np.log(feature_19+1) if feature_19>=0 else -np.log(abs(feature_19)+1)
    feature_20 = np.log(feature_20+1) if feature_20>=0 else -np.log(abs(feature_20)+1)
    feature_21 = np.log(feature_21+1) if feature_21>=0 else -np.log(abs(feature_21)+1)
    feature_22 = np.log(feature_22+1) if feature_22>=0 else -np.log(abs(feature_22)+1)
    feature_23 = np.log(feature_23+1) if feature_23>=0 else -np.log(abs(feature_23)+1)
    feature_24 = np.log(feature_24+1) if feature_24>=0 else -np.log(abs(feature_24)+1)
    feature_25 = np.log(feature_25+1) if feature_25>=0 else -np.log(abs(feature_25)+1)
    feature_26 = np.log(feature_26+1) if feature_26>=0 else -np.log(abs(feature_26)+1)
    feature_27 = np.log(feature_27+1) if feature_27>=0 else -np.log(abs(feature_27)+1)
    feature_28 = np.log(feature_28+1) if feature_28>=0 else -np.log(abs(feature_28)+1)
    feature_29 = np.log(feature_29+1) if feature_29>=0 else -np.log(abs(feature_29)+1)
    feature_30 = np.log(feature_30+1) if feature_30>=0 else -np.log(abs(feature_30)+1)
    feature_31 = np.log(feature_31+1) if feature_31>=0 else -np.log(abs(feature_31)+1)
    feature_32 = np.log(feature_32+1) if feature_32>=0 else -np.log(abs(feature_32)+1)
    feature_33 = np.log(feature_33+1) if feature_33>=0 else -np.log(abs(feature_33)+1)
    feature_34 = np.log(feature_34+1) if feature_34>=0 else -np.log(abs(feature_34)+1)
    feature_35 = np.log(feature_35+1) if feature_35>=0 else -np.log(abs(feature_35)+1)
    feature_36 = np.log(feature_36+1) if feature_36>=0 else -np.log(abs(feature_36)+1)

    this_feature = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, \
        feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, \
        feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23,\
        feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, \
        feature_31, feature_32,feature_33, feature_34,feature_35,feature_36] 

    this_feature_graph = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, \
        feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, \
        feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22,\
        feature_25, feature_26, feature_27, feature_28, feature_30, \
        feature_32, feature_33]   

    data_select_all.append(this_feature)
    data_graph_all.append(this_feature_graph)
    data_target_all.append(muchan)


data_select = DataFrame(data_select_all)
data_graph = DataFrame(data_graph_all)
data_target = data_target_all

all_features = pd.get_dummies(data_select, dummy_na=True)
graph_features = pd.get_dummies(data_graph, dummy_na=True)

all_features = np.nan_to_num(all_features)
graph_features = np.nan_to_num(graph_features)


print("Applying Standardization to features...")

scaler_feat = StandardScaler()
scaler_graph = StandardScaler()

#Z-score normalization
all_features = scaler_feat.fit_transform(all_features)


graph_features = scaler_graph.fit_transform(graph_features)


features_tensor = torch.tensor(all_features, dtype=torch.float)
graph_tensor = torch.tensor(graph_features, dtype=torch.float)
labels_tensor = torch.tensor(data_target, dtype=torch.float)

print(f"Total samples: {features_tensor.shape[0]}")
    
def Eu_dis(x):
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def _generate_G_from_H(H, variable_weight=True):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    n_obj = dis_mat.shape[0] 
    n_edge = n_obj    
    H = np.zeros((n_obj, n_edge))    
    for center_idx in range(n_obj): 
        dis_mat[center_idx, center_idx] = 0 
        dis_vec = dis_mat[center_idx] 
        res_vec = np.argsort(dis_vec.A)
        nearest_idx = res_vec[0]
        avg_dis = np.average(dis_vec) 
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH: 
                H[center_idx,node_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[center_idx,node_idx] = 1.0
    return H

class HGNN_conv(nn.Module):
    
    def __init__(self, in_ft, out_ft, bias=True, k_neig=5):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.k_neig = k_neig  
        
        self.a = nn.Parameter(torch.zeros(size=(self.k_neig * out_ft, 1)))
        self.leakyrelu = nn.LeakyReLU()
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.a.data, -stdv, stdv)    

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        
        edge_4att = x.matmul(self.weight)
    
       
        pair = G.nonzero().t().cuda() 
        N1 = G.shape[0] 
        N2 = G.shape[1]
        
        
        qq = G.nonzero().t()[1]
        y = edge_4att[qq] 
        
        
        
        if y.shape[0] % self.k_neig != 0:
           
            valid_len = (y.shape[0] // self.k_neig) * self.k_neig
            y = y[:valid_len]
            
            pair = pair[:, :valid_len]

        
        yy = y.reshape(y.shape[0] // self.k_neig, y.shape[1] * self.k_neig).cuda()
        pair_h = yy

      
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        pair_ee = pair_e.cpu().detach().numpy()
        
     
        pair_e2 = pair_ee.repeat(self.k_neig)
        pair_e = t.tensor(pair_e2).cuda()
        
        pair_e = F.softmax(pair_e, dim=0) 
        
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([N1, N2])).to_dense()
        e = e + G 
        
        if self.bias is not None:
            edge_4att = edge_4att + self.bias
        x = e.matmul(edge_4att)
        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, k_neig=5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.k_neig = k_neig
        
       
        self.hgc1 = HGNN_conv(in_ch, n_hid, k_neig=self.k_neig) 
        self.hgc2 = HGNN_conv(n_hid, n_hid, k_neig=self.k_neig)
        self.hgc3 = HGNN_conv(n_hid, n_hid//2, k_neig=self.k_neig)
        self.hgc4 = HGNN_conv(n_hid//2, n_hid//4, k_neig=self.k_neig)
        
      
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid//2)
        self.bn4 = nn.BatchNorm1d(n_hid//4)
        
       
        self.out_layer = nn.Sequential(
            nn.Linear(n_hid//4, n_hid//8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid//8, n_class)
        )

    def forward(self, x, G):
       
        x1 = self.hgc1(x, G)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        
        
        x2 = self.hgc2(x1, G)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training)
        
      
        x3 = self.hgc3(x2, G)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=self.training)
        
       
        x4 = self.hgc4(x3, G)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        
        
        out = self.out_layer(x4)
        return out

def main():
    print("Constructing Graph for all data...")
    x = Eu_dis(graph_tensor)
    H = construct_H_with_KNN_from_distance(x, 5, False, 1) 
    G = np.nan_to_num(H)
    G = torch.Tensor(G).cuda()
    
    features_all = features_tensor.cuda()
    labels_all = labels_tensor.cuda().view(-1, 1)
    
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
   
    fold_metrics = {
        'Fold': [],
        'R2': [],
        'MSE': [],
        'RMSE': [],
        'MAE': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(features_all)):
        print(f"\n================ Fold {fold + 1} / {k_folds} ================")
        
        train_idx_tensor = torch.tensor(train_idx).cuda()
        test_idx_tensor = torch.tensor(test_idx).cuda()
        
        t.manual_seed(0)

    net = HGNN(36, 1, 128, dropout=0.3, k_neig=5).cuda() 
        
      
    optimizer = t.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        
       
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
    
    criterion = nn.HuberLoss(delta=1.0) 
        
    best_rmse = float('inf') 
        scaler = GradScaler()
        
        epochs = 15000 
        
        for epoch in range(epochs):
            net.train()
            optimizer.zero_grad()
            
            with autocast():
                outputs = net(features_all, G)
                output = outputs.float()
                loss = nn.MSELoss()(output[train_idx_tensor], labels_all[train_idx_tensor])
            
            if loss < 0.001:
                break
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

           
            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                   
                    test_out = net(features_all, G).float()
                    curr_test_loss = nn.MSELoss()(test_out[test_idx_tensor], labels_all[test_idx_tensor])
                    curr_rmse = torch.sqrt(curr_test_loss).item()
                    
                  
                    if curr_rmse < best_rmse:
                        best_rmse = curr_rmse
                       
                net.train()

      
        net.eval()
        with torch.no_grad():
            final_output = net(features_all, G).float()
            test_pred = final_output[test_idx_tensor].detach().cpu().numpy().flatten()
            test_real = labels_all[test_idx_tensor].detach().cpu().numpy().flatten()
            
            r2 = r2_score(test_real, test_pred)
            mse = mean_squared_error(test_real, test_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_real, test_pred)
            
            fold_metrics['Fold'].append(f'Fold {fold + 1}')
            fold_metrics['R2'].append(r2)
            fold_metrics['MSE'].append(mse)
            fold_metrics['RMSE'].append(rmse)
            fold_metrics['MAE'].append(mae)
            
            print(f"Fold {fold + 1} Result | R2: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
            
           
            result_df = pd.DataFrame({
                'Real Labels': test_real,
                'Predictions': test_pred
            })
            result_df.to_csv(f'fold_{fold+1}_predictions_成熟期_k=5_改进.csv', index=False)

   
    print("\n================ Cross Validation Results ================")
    
   
    metrics_df = pd.DataFrame(fold_metrics)
    
 
    avg_metrics = metrics_df.iloc[:, 1:].mean()
    std_metrics = metrics_df.iloc[:, 1:].std()
    
   
    avg_row = pd.DataFrame([['Average'] + avg_metrics.tolist()], columns=metrics_df.columns)
    std_row = pd.DataFrame([['Std'] + std_metrics.tolist()], columns=metrics_df.columns)
   
    final_metrics_df = pd.concat([metrics_df, avg_row, std_row], ignore_index=True)
    
   
    print(f"Average R2:   {avg_metrics['R2']:.4f} ± {std_metrics['R2']:.4f}")
    print(f"Average MSE:  {avg_metrics['MSE']:.4f} ± {std_metrics['MSE']:.4f}")
    print(f"Average RMSE: {avg_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}")
    print(f"Average MAE:  {avg_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f}")
    
 
    output_csv_name = 'result.csv'
    final_metrics_df.to_csv(output_csv_name, index=False)
    print(f"\nEvaluation metrics have been saved to '{output_csv_name}'")

if __name__ == "__main__":                                    
    main()
