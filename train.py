import gensim
import re
import logging
import os


class MyCorpus:
    def __iter__(self):
        for i in ['RS232-T1000', 'RS232-T1100', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500',
                  'RS232-T1600']:
            # for j in ['truelabel.txt','falselabel.txt']:
            #         corpus_path = os.path.join(os.getcwd(),i,j)
            #         print(i,j)
            #         for line in open(corpus_path):
            #             # assume there's one document per line, tokens separated by whitespace
            #              yield simple_preprocess(line)
            for j in range(2, 7):
                path = os.path.join('dataset', '{}_level{}_text.txt'.format(i, j))
                for line in open(path):
                    yield line.split()


logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
sentences = MyCorpus()
# for v in sentences:
#     print(v)
model = gensim.models.Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
                               workers=8, epochs=10)
# model.wv.save_word2vec_format("model.bin", binary=True)
model.wv.save_word2vec_format('data_test/word2vec.txt')
