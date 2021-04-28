import pandas as pd
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import torch

def csv_spilit():
    data = pd.read_csv("data/data.csv", ',', error_bad_lines=False)  # 我的数据集是两列，一列字符串，一列为0,1的label
    data = np.array(data)
    random.shuffle(data)  # 随机打乱
    # 取前90%为训练集
    print('开始处理......')
    allurl_fea = [d[0] for d in data]
    df1 = data[:int(0.999 * len(allurl_fea))]
    # 将np.array转为dataframe，并对两列赋列名
    df1 = pd.DataFrame(df1, columns=['text', 'label'])
    # 写入csv
    df1.to_csv("data/train_samples.csv", index=False)
    # 剩余百分之10为训练集
    df2 = data[int(0.999 * len(allurl_fea)):]
    df2 = pd.DataFrame(df2, columns=['text', 'label'])
    df2.to_csv("data/test_samples.csv", index=False)
    # 由于我的数据集中是二分类的，检测下两个类别分别的占比
    print(df1['label'].value_counts())
    print('处理完成')


def csv_merge(path, destination):
    csv_list = glob.glob(path)  # 查看同文件夹下的csv文件数
    print(u'共发现%s个CSV文件' % len(csv_list))
    print(csv_list)
    print(u'正在处理............')
    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(csv_list[0])
    df.to_csv(destination, encoding="utf_8_sig", index=False)
    for i in range(1, len(csv_list)):  # 循环读取同文件夹下的csv文件
        df = pd.read_csv(csv_list[i])
        df.to_csv(destination, encoding="utf_8_sig", index=False, header=False, mode='a+')
    print(u'合并完毕！')


def samples_equal_split_process(source):
    csv_list = glob.glob(source)  # 查看同文件夹下的csv文件数
    print(u'共发现%s个CSV文件' % len(csv_list))
    df = pd.read_csv(csv_list[0])
    data = np.array(df)
    # 处理负样本
    mylist1 = []
    for d in data:
        if (d[1] == 1):
            mylist1.append(d)
    df1 = np.array(mylist1)
    df1 = pd.DataFrame(df1, columns=['text', 'label'])
    df1.to_csv('data/tmp/negtive_samples.csv', index=False)
    # 处理正样本
    mylist2 = []
    random.shuffle(data)
    count = 0
    for d in data:
        if (d[1] == 0):
            count += 1
            mylist2.append(d)
    # 进行正样本抽样
    data_sample = random.sample(mylist2, len(mylist1))
    print(len(data_sample))
    df2 = np.array(data_sample)
    df2 = pd.DataFrame(df2, columns=['text', 'label'])
    df2.to_csv('data/tmp/postive_samples.csv', index=False)

    # 循环追加处理正负样本
    for i in range(1, len(csv_list)):
        df = pd.read_csv(csv_list[i])
        data = np.array(df)
        # 处理负样本
        mylist1 = []
        for d in data:
            if (d[1] == 1):
                mylist1.append(d)
        df1 = np.array(mylist1)
        df1 = pd.DataFrame(df1, columns=['text', 'label'])
        df1.to_csv('data/tmp/negtive_samples.csv', index = False, header = False , mode = 'a+')
        # 处理正样本
        mylist2 = []
        random.shuffle(data)
        count = 0
        for d in data:
            if (d[1] == 0):
                count += 1
                mylist2.append(d)

        # 进行正样本抽样
        data_sample = random.sample(mylist2, len(mylist1))
        print(len(data_sample))
        df2 = np.array(data_sample)
        df2 = pd.DataFrame(df2, columns=['text', 'label'])
        df2.to_csv('data/tmp/postive_samples.csv', index=False, header=False, mode='a+')



def postive_samples_process():
    df = pd.read_csv("data/data.csv")
    data = np.array(df)
    random.shuffle(data)
    count = 0
    mylist1 = []
    for d in data:
        # print(isinstance(d[1],int))
        tmp = d[1]
        if (tmp == 0 and count < 625226):
            count += 1
            mylist1.append(d)
    df1 = np.array(mylist1)
    df1 = pd.DataFrame(df1, columns=['text', 'label'])
    df1.to_csv('data/postive_samples.csv', index=False)

def generate_equal_data(source, destination):
    samples_equal_split_process(source=source)
    csv_merge(path='data/tmp/*.csv', destination=destination)
def generate_normal_data():
    csv_merge(path='dataset/tmp/*.csv', destination='dataset/data_equal/data6.csv')

def shuffle_csv():

    # df3 = pd.read_csv("test_samples.csv")
    # df0 = df3.sample(frac=1)
    # df0.to_csv("test_samples.csv", index=False)
    df1 = pd.read_csv("data_test/data_equal/data.csv")
    df2 = pd.read_csv("data_test/data_equal/data6.csv")
    for i in range(1, 30):
        df1 = df1.sample(frac=1)
        df2 = df2.sample(frac=1)
    df1.to_csv("data_test/data_equal/data.csv", index=False)
    df2.to_csv("data_test/data_equal/data6.csv", index=False)

# generate_equal_data(source='data/level_6/*.csv', destination='dataset/data_equal/data4.csv')
# generate_normal_data()
# generate_normal_data()
shuffle_csv()