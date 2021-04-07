import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        # self.args = args
        # model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary= True)
        model = gensim.models.Word2Vec.load('word2vec.model')
        # print(model.wv.vectors)
        weights  = torch.FloatTensor(model.wv.vectors)
        print(weights.shape)
        embedding = nn.Embedding.from_pretrained(weights)
        query = 'emp'
        query_id  = torch.tensor(model.wv.vocab['emp'].index)
        gensim_vec = torch.tensor(model[query])
        embedding_vec = embedding(query_id)
        # print(gensim_vec)
        print("done init")
    def forward(self,x): 
        return x

m = TextCNN()
