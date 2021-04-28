from torchtext.legacy import data
from torchtext.vocab import Vectors
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os
import re
import jieba
import scipy.io as io
from torchtext.vocab import GloVe
import psutil

class MyDataset(data.Dataset):

    def __init__(self, datas, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("text", text_field), ("label", label_field)]

        examples = []
        # csv_data = pd.read_csv(path)
        # print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(datas['text']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(datas['text'], datas['label'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([text, label], fields))
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


def tokenizer(text):  # create a tokenizer function
    return text


# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


def load_data(args, train_data, valid_data, test_data):
    print('加载数据中...')
    # stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.ma_len, stop_words=stop_words)
    '''

    # Field定义怎么处理数据
    TEXT = data.Field(sequential=True, lower=False)
    LABEL = data.Field(sequential=False)

    # dataset
    '''
    train, val = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='train_samples.csv',
            validation='test_samples.csv',
            format='csv',
            fields=[('text', text),('label', label)],#告诉fields处理哪些数据+Field怎么处理数据
        )
    '''

    train = MyDataset(train_data, text_field=TEXT, label_field=LABEL, test=False)

    val = MyDataset(valid_data, text_field=TEXT, label_field=LABEL, test=False)

    test = MyDataset(test_data, text_field=TEXT, label_field=LABEL, test=False)  # 这边测试集也加载label

    # 构建词表
    if args.static:
        TEXT.build_vocab(train, vectors=Vectors(name='data_test/word2vec.txt'))  # 此处改为你自己的词向量
        print(TEXT.vocab.stoi)
        print("加载的词向量:", TEXT.vocab.vectors, TEXT.vocab.vectors.size())
        args.embedding_dim = TEXT.vocab.vectors.size()[-1]
        args.vectors = TEXT.vocab.vectors
    else:
        TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    if args.cuda:
        run_device = torch.device('cuda')
    else:
        run_device = torch.device('cpu')
    # 设置迭代器
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.text),
        batch_sizes=(args.batch_size, 1024, 1024),  # 训练集设置batch_size,验证集整个集合用于测试
        device=run_device
    )
    args.vocab_size = len(TEXT.vocab)
    args.label_num = len(LABEL.vocab)
    return train_iter, val_iter, test_iter

def showMem():
    mem = psutil.virtual_memory()
    print('剩余内存{:.6f}GB'.format(float(mem.free/1024/1024/1024)))
