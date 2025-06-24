import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
# import torch.cuda.amp as amp
torch.set_default_tensor_type(torch.FloatTensor)
from torch.cuda.amp import autocast, GradScaler ########
from torch.autograd import Variable
from pandas.core.frame import DataFrame
from networkx import karate_club_graph,to_numpy_array  #from networkx import karate_club_graph,to_numpy_matrix
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
from pytorch_metric_learning import losses
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_dense_adj
from collections import Counter
from torch.nn.parameter import Parameter
import math 
import os




corn_file=open('train_test_output.csv',encoding='gbk')   
csv_reader_lines = csv.reader(corn_file)   

data_select_train = []
data_select_test  = []
data_graph_train  = []
data_graph_test   = []
data_target_train = []
data_target_test  = []
k = 0
num_train = 0
num_test = 0
num = 0


for row in csv_reader_lines: 

    if num == 0:
        num = num+1
        information = row
        continue

    if num >=10001:
        break


    num = num + 1
    


    muchan = float(row[37])    
    feature_1 = float(row[1])
    feature_2 = float(row[2])
    feature_3 = float(row[3])
    feature_4 = float(row[4])
    feature_5 = float(row[5])
    feature_6 = float(row[6])
    feature_7 = float(row[7])
    feature_8 = float(row[8])
    feature_9 = float(row[9])
    feature_10 = float(row[10])
    feature_11 = float(row[11])
    feature_12 = float(row[12])
    feature_13 = float(row[13])
    feature_14 = float(row[14])
    feature_15 = float(row[15])
    feature_16 = float(row[16])
    feature_17 = float(row[17])
    feature_18 = float(row[18])
    feature_19 = float(row[19])
    feature_20 = float(row[20])
    feature_21 = float(row[21])
    feature_22 = float(row[22])
    feature_23 = float(row[23])
    feature_24 = float(row[24])
    feature_25 = float(row[25]) 
    feature_26 = float(row[26])
    feature_27 = float(row[27])
    feature_28 = float(row[28])
    feature_29 = float(row[29])
    feature_30 = float(row[30])
    feature_31 = float(row[31])
    feature_32 = float(row[32])
    feature_33 = float(row[33])
    feature_34 = float(row[34])
    feature_35 = float(row[35]) 
    feature_36 = float(row[36])



    
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

    if num<=8000:  
        data_select_train.append(this_feature) 
        data_graph_train.append(this_feature_graph) 
        data_target_train.append(muchan) 
        num_train = num_train + 1

    if num>8000:  
        data_select_test.append(this_feature) 
        data_graph_test.append(this_feature_graph)
        data_target_test.append(muchan)
        num_test= num_test + 1   


data_select_train = DataFrame(data_select_train)
data_graph_train = DataFrame(data_graph_train)
data_select_test = DataFrame(data_select_test)
data_graph_test = DataFrame(data_graph_test)

data_select = pd.concat ([data_select_train,data_select_test],axis=0)
data_graph = pd.concat([data_graph_train,data_graph_test],axis=0)    
data_target = data_target_train + data_target_test      

all_features = pd.get_dummies(data_select, dummy_na=True)
graph_features = pd.get_dummies(data_graph, dummy_na=True)


all_features = np.nan_to_num(all_features)
graph_features = np.nan_to_num(graph_features)
try2 = np.array([[1,1],[1,2],[2,2],[2,3],[1,0]])
train_try = torch.tensor(try2, dtype=torch.float)
train_features = torch.tensor(all_features, dtype=torch.float)
train_graph = torch.tensor(graph_features, dtype=torch.float)
train_labels = torch.tensor(data_target, dtype=torch.float)
    
def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def _generate_G_from_H(H, variable_weight=True):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
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
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1): 
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
   
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
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.a = nn.Parameter(torch.zeros(size=(5*out_ft, 1)))
        self.leakyrelu = nn.LeakyReLU()
        self.G2 = torch.eye(10000).cuda()  
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
        pair_h = torch.rand(10001,x.shape[1]*5).cuda()   
        qq=G.nonzero().t()[1]
        y = edge_4att[qq] 
        yy = y.reshape(y.shape[0]//5,y.shape[1]*5).cuda()   
        pair_h = yy

       
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        pair_ee = pair_e.cpu().detach().numpy()
        pair_e2 = pair_ee.repeat(5)
        pair_e = t.tensor(pair_e2).cuda()    
        pair_e = F.softmax(pair_e, dim =0) // 1000
        assert not torch.isnan(pair_e).any()
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([N1, N2])).to_dense()
        e = e + G 
        
        if self.bias is not None:
            edge_4att = edge_4att + self.bias
       
        x = e.matmul(edge_4att)
        return x 

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid//2)
        self.hgc2 = HGNN_conv(n_hid//2, n_hid//4)
        self.hgc3 = HGNN_conv(n_hid//4, n_hid//8)
        self.hgc4 = HGNN_conv(n_hid//8, n_hid//16)
        self.hgc5 = Linear(n_hid//16, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        
        x = F.relu(self.hgc2(x, G))
        
        x = F.relu(self.hgc3(x, G))
        
        x = F.relu(self.hgc4(x, G))
        
        x = self.hgc5(x)
        return x




       

def main():
    # 截取数据
    train_data = train_features
    train_label = train_labels
    print("Train:",train_data.shape) 
    MAE_Label = []
    MSE_Label = []
    RMSE_Label = []
    Loss_Label = []
    x = Eu_dis(train_graph)
   
    H = construct_H_with_KNN_from_distance(x, 4,False, 1)  ##    H = 
    G = np.nan_to_num(H)
    G = torch.Tensor(G).cuda()     


   
   
    labels = t.tensor([i for i in range(8000)]).cuda()     
    labels_test = [i+8000 for i in range(2000)] 
    t.manual_seed(0)
    net = HGNN(36, 1, 256).cuda() 
    

    lr = 0.015
    optimizer = t.optim.Adam(net.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,2000,4000,6000,10000,15000,20000], gamma=0.9)
    
    
    train_data = train_data.cuda()  



    predictions = []
    labels_actual = []



    

   
    scaler = GradScaler()

    for epoch in range(30000):
        optimizer.zero_grad()
       
        with autocast(): 
            outputs = net(train_data,G) 
            output = outputs.float() 
            labels = train_labels.cuda().view(-1, 1) 
            criterion = nn.MSELoss()
            loss = criterion(output, labels)  
           
        if loss < 0.001:
            break
       
        optimizer.zero_grad()

        
        scaler.scale(loss).backward()  

       
        scaler.step(optimizer)
        scaler.update() 

        scheduler.step()

        n_output = output.detach().cpu().numpy().reshape(10000)
        n_label = labels.detach().cpu().numpy()

        predictions.extend(n_output)  
        labels_actual.extend(n_label) 

        R2 =  r2_score(n_output[labels_test],n_label[labels_test])
        mse = mean_squared_error(n_output[labels_test],n_label[labels_test])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(n_output[labels_test],n_label[labels_test])
        
        print('\nEpoch %d | Loss: %.4f | R2: %.4f | MSE: %.4f | RMSE: %.4f | MAE %.4f ' % (epoch, loss.item(), R2, mse, rmse, mae)) 
        MAE_Label.append(mae)
        MSE_Label.append(mse)
        RMSE_Label.append(rmse)
        Loss_Label.append(R2)
    min_MAE = min(MAE_Label)
    min_MSE = min(MSE_Label)
    min_RMSE = min(RMSE_Label)
    R22 = max(Loss_Label)
    print('\n maxR2: %.4f | minMSE: %.4f | minRMSE: %.4f | minMAE %.4f ' % ( R22, min_MSE, min_RMSE, min_MAE))
    
    
    
  
    result_df = pd.DataFrame({
        'Real Labels': labels_actual,
        'Predictions': predictions
    })

  
    result_df.to_csv('predictions_vs_real_labels10000-4-36.csv', index=False)
    print("Predictions and real labels have been saved to 'predictions_vs_real_labels10000-4-36.csv'.")     

if __name__ == "__main__":                                                
    main()