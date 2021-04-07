import gensim
import re
import logging
import os
class MyCorpus:
    def __iter__(self):
        for i in ['RS232-T1000','RS232-T1100','RS232-T1200','RS232-T1300','RS232-T1400','RS232-T1500','RS232-T1600']:
        #     for j in ['truelabel.txt','falselabel.txt']:
        #         corpus_path = os.path.join(os.getcwd(),i,j)
        #         print(i,j)
        #         for line in open(corpus_path):
        #             # assume there's one document per line, tokens separated by whitespace
        #              yield simple_preprocess(line)
        # for line in open('ans.txt'):
        #     yield re.split('[, ]',line)
            path = os.path.join(i,'text.txt')
            for line in open(path):
                yield gensim.utils.simple_preprocess(line)
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
sentences = MyCorpus()
# for v in sentences:
    # print(v)
model=gensim.models.Word2Vec(sentences,sg=1,vector_size=100,window=5,min_count=2,negative=3,sample=0.001,hs=1,workers=4, epochs=10)
# model.wv.save_word2vec_format("model.bin", binary=True)
model.wv.save_word2vec_format('word2vec.vector')
# import torchtext.vocab as Vocab
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gensim
# W2V_TXT_FILE="model.vector"
# W2V_BIN_FILE="model.bin"
# CACHE_DIR="cache"
# model = gensim.models.KeyedVectors.load_word2vec_format(W2V_BIN_FILE,binary=True)
# # model.wv.save_word2vec_format(W2V_TXT_FILE)
# vectors = Vectors(name=W2V_TXT_FILE,cache = CACHE_DIR)
# weights = torch.FloatTensor(vectors.vectors)
# Embed_dim= weights.size(1) #(words, dim)
# vocnum =weights.size(0) # total word number
# print(weights)
# print(Embed_dim)
# print(vocnum)