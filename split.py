import pandas as pd
import numpy as np
import random
import os
import glob
def csv_spilit():
    data = pd.read_csv("RS232-T1000/data.csv",',',error_bad_lines=False)#我的数据集是两列，一列字符串，一列为0,1的label
    data = np.array(data)
    random.shuffle(data)#随机打乱
    #取前90%为训练集
    print('开始处理......')
    allurl_fea = [d[0] for d in data]
    df1=data[:int(0.999*len(allurl_fea))]
    #将np.array转为dataframe，并对两列赋列名
    df1=pd.DataFrame(df1,columns=['text','label'])
    #写入csv
    df1.to_csv("data/train_samples.csv",index=False)
    #剩余百分之10为测试集
    df2=data[int(0.999*len(allurl_fea)):]
    df2=pd.DataFrame(df2,columns=['text','label'])
    df2.to_csv("data/test_samples.csv",index=False)
    #由于我的数据集中是二分类的，检测下两个类别分别的占比
    print(df1['label'].value_counts())
    print('处理完成')
def csv_merge():
    csv_list = glob.glob('data/*.csv') #查看同文件夹下的csv文件数
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(csv_list)
    print(u'正在处理............')
    #读取第一个CSV文件并包含表头
    df = pd.read_csv(csv_list[0])
    df.to_csv('data.csv',encoding="utf_8_sig",index=False)
    for i in range(1,len(csv_list)): #循环读取同文件夹下的csv文件
        df = pd.read_csv(csv_list[i])
        df.to_csv('data.csv',encoding="utf_8_sig",index=False, header=False, mode='a+')
    print(u'合并完毕！')
csv_spilit()