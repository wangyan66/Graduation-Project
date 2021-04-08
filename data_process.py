from torchtext.legacy import data
from torchtext.vocab import Vectors
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os
import re
import jieba
class MyDataset(data.Dataset):

    def __init__(self, data, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("text", text_field), ("label", label_field)]
        
        examples = []
        # csv_data = pd.read_csv(path)
        # print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(data['text']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(data['text'], data['label'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([None, text, label], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super(MyDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)

def tokenizer(text): # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

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
        
    return X_train, X_valid

def load_data(args,train_data,valid_data):
    print('加载数据中...')
    # stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    #Field定义怎么处理数据
    TEXT = data.Field(sequential=True, lower=True)
    LABEL = data.Field(sequential=False)
    
    
    #dataset
    train, val = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='train_samples.csv',
            validation='test_samples.csv',
            format='csv',
            fields=[('text', text),('label', label)],#告诉fields处理哪些数据+Field怎么处理数据
        )
    train = MyDataset(train_data, text_field=TEXT, label_field=LABEL, test=False)
    valid = MyDataset(valid_data, text_field=TEXT, label_field=LABEL, test=False)
    #构建词表
    if args.static:
        text.build_vocab(train, val, vectors=Vectors(name="data/model.vector")) # 此处改为你自己的词向量
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors

    else: text.build_vocab(train, val)

    label.build_vocab(train, val)
    if args.cuda:
        run_device= torch.device('cuda')
    #设置迭代器
    train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(args.batch_size, 128), # 训练集设置batch_size,验证集整个集合用于测试
            device=run_device
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    return train_iter, val_iter

# import torchtext.vocab as Vocab
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gensim
# W2V_TXT_FILE="model.vector"
# W2V_BIN_FILE="model.bin"
# CACHE_DIR="cache"
# # text = data.Field(sequential=True, lower=True)
# # label = data.Field(sequential=False)
# model = gensim.models.KeyedVectors.load_word2vec_format(W2V_BIN_FILE,binary=True)
# # model.wv.save_word2vec_format(W2V_TXT_FILE)
# vectors = Vectors(name=W2V_TXT_FILE,cache = CACHE_DIR)
# weights = torch.FloatTensor(vectors.vectors)
# Embed_dim= weights.size(1) #(words, dim)
# vocnum =weights.size(0) # total word number
# print(weights)
# print(Embed_dim)
# print(vocnum)