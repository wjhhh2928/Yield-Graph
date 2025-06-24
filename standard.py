import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv 
import os

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


data = pd.read_csv("train_test.csv",encoding='gbk')
print(data.shape)
data = data.values

point = np.where(data==-0.001)
data2 = data[:,0:41]
print(data2.shape)

this_data = data2.astype('float')
this_data2 = standardization(this_data)


pd.DataFrame(this_data2).to_csv('standard_data_new.csv')