import numpy as np
import torch
import torch.nn.functional as F
import pickle
import torch.nn as nn

import sys
sys.path.append('/home/zcy/zyq/grape_y')


from ..models.gnn_model import get_gnn    ##原：from ..models.gnn_model import get_gnn
from ..models.prediction_model import MLPNet2, MLPNet   ##原：from ..models.prediction_model import MLPNet2, MLPNet 
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge
from matplotlib import pyplot as plt


class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        # N = (x.size(0) * x.size(1))
        N = (x.size(0) * 1)

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])

class GHMR_Loss(GHM_Loss):
    # 回归损失
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        device=torch.device('cuda')
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * 1
        weight = weight.to(device)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)

def train_gnn_y(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)
    n_row, n_col = data.df_X.shape
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    impute_model = MLPNet(input_dim, 1,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
    n_row, n_col = data.df_X.shape
    predict_model = MLPNet2(n_col, 64,1).to(device)                      

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask*all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
        print("all y num is {}, train num is {}, valid num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(valid_y_mask),torch.sum(test_y_mask)))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_y_mask = all_train_y_mask.clone().detach()
        print("all y num is {}, train num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(test_y_mask)))
    best = 1000

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        predict_model.train()
        index = int(train_edge_attr.shape[0]/2)
        known_mask = get_known_mask(train_edge_attr[0:index]).to(device)
        # known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])
        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss_func = GHMR_Loss(bins=10, alpha=0.75, mu = 1 )
        loss2 = loss_func(pred_train, label_train)
        # loss = F.mse_loss(pred_train, label_train) + loss2
        # loss = F.mse_loss(pred_train, label_train)
        loss = loss2
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        predict_model.eval()
        with torch.no_grad():
            x_embd = model(x, train_edge_attr, train_edge_index)
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            X = torch.reshape(X, [n_row, n_col])
            pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            mse = F.mse_loss(pred_test, label_test)

            if epoch == 60000:
                x1 = [1,1000] #两个点
                y1 = [1,1000]
                plt.plot(x1,y1,color='y')
                plt.scatter(label_test.cpu(), pred_best.cpu() ,c='none',s=4, marker='o',edgecolors='b',alpha = 2/5, label="True") #颜色表示
                plt.axis([250,900,250,900])#设定x轴 y轴的范围

                plt.legend()
                plt.savefig("./"+str(epoch)+".png")
  

            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()

            Train_loss.append(train_loss)
            Test_rmse.append(test_rmse)
            Test_l1.append(test_l1)
            print('epoch: ', epoch)
            print('loss: ', train_loss)
           
            print('test rmse: ', test_rmse)
            print('test l1: ', test_l1)
            if best > l1:
                l1 = best
                pred_best = pred_test
    print('min test rmse: ', np.min(Test_rmse))
    print('min test l1: ', np.min(Test_l1))

    # xx = np.arange(1,3903)
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.scatter(xx, label_test.cpu() ,c='none',s=2, marker='o',edgecolors='r',alpha = 2/5, label="True") #颜色表示
    # plt.scatter(xx, pred_best.cpu() ,c='none',s=2, marker='o',edgecolors='g',alpha = 2/5,label="Pre") 
 
    # plt.xlabel("number") #x轴命名表示
    # plt.ylabel("yield") #y轴命名表示
    # plt.axis([0,4000,250,1200])#设定x轴 y轴的范围
    # plt.title("Ours")
    # plt.legend()
    # plt.savefig("./"+"ours"+".png")
#    #####################################################
    x = [1,1000] #两个点
    y = [1,1000]
    plt.plot(x,y,color='y')
    
    plt.scatter(label_test.cpu(), pred_best.cpu() ,c='none',s=4, marker='o',edgecolors='b',alpha = 2/5, label="True") #颜色表示

    plt.xlabel("True") #x轴命名表示
    plt.ylabel("Pre") #y轴命名表示

    plt.axis([250,900,250,900])#设定x轴 y轴的范围
    plt.title("Ours")
    plt.legend()
    plt.savefig("./"+"Ours22"+".png")

    