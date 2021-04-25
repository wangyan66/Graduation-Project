# 导入模块
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
def get_kfold_data(k, i, X, y):  
     
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]     # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        
    return X_train, y_train, X_valid,y_valid
# 创建一个数据集
# X = torch.rand(500, 100, 10)
# Y = torch.rand(500, 1)
# # X = X.view(X.size(0),X.size(1), X.size(2),1)
# m  = nn.Conv1d(15,100,3)
# out = m(X)
# print(out)
# X.view()
print(range(1,10))
haha = [i for i in range(1,10)]
print(haha)
isinstance(haha,list)
# x = torch.rand((2,2,3))
# y = torch.rand((2,2,3))
# print("x:",x)
# print("y:",y)
# print("dim=0:", torch.cat((x,y),dim=0).size())
# print("dim=1:", torch.cat((x,y), dim=1).size())
# print("dim=2:", torch.cat((x, y), dim=2).size())