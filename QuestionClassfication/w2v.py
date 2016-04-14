import gensim
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(cnn_train, cnn_vocabulary_inv,lstm_train,lstm_vocabulary_inv):
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    cnn_embedding_weights = [np.array([model[w] if w in model\
                                                        else np.random.uniform(-0.25,0.25,300)\
                                                        for w in cnn_vocabulary_inv])]
    lstm_embedding_weights = [np.array([model[w] if w in model\
                                                        else np.random.uniform(-0.25,0.25,300)\
                                                        for w in lstm_vocabulary_inv])]
    return [cnn_embedding_weights,lstm_embedding_weights]

if __name__=='__main__':
    import data_helpers
    print("Loading data...")
    cnn_train,lstm_train,Y_train,cnn_vocabulary,cnn_vocabulary_inv,lstm_vocabulary,lstm_vocabulary_inv= data_helpers.load_data()
    cnn_embedding_weights, lstm_embedding_weights= train_word2vec(cnn_train, cnn_vocabulary_inv,lstm_train,lstm_vocabulary_inv)
    print("success")

