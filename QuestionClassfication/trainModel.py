import numpy as np
import keras
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
import data_helpers
from w2v import train_word2vec
from sklearn.cross_validation import StratifiedKFold

cnn_train,lstm_train,Y_train,cnn_vocabulary,cnn_vocabulary_inv,lstm_vocabulary,lstm_vocabulary_inv = data_helpers.load_data()
cnn_embedding_weights,lstm_embedding_weights = train_word2vec(cnn_train, cnn_vocabulary_inv,lstm_train,lstm_vocabulary_inv)
#cnn_train=cnn_embedding_weights[0][cnn_train]
#lstm_train=lstm_embedding_weights[0][lstm_train]

shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
cnn_shuffled = cnn_train[shuffle_indices]
lstm_shuffled = lstm_train[shuffle_indices]
Y_train = Y_train[shuffle_indices]
#Y_train_f=np_utils.to_categorical(Y_train,27)

filter_sizes = (3, 4)
num_filters = 150
hidden_dims = 150

cnn_graph = Graph()
cnn_graph.add_input(name='input', input_shape=(32, 300))
for fsz in filter_sizes:
	conv = Convolution1D(nb_filter=num_filters,filter_length=fsz,border_mode='valid',activation='relu',subsample_length=1)
	pool = MaxPooling1D(pool_length=2)
	cnn_graph.add_node(conv, name='conv-%s' % fsz, input='input')
	cnn_graph.add_node(pool, name='maxpool-%s' % fsz, input='conv-%s' % fsz)
	cnn_graph.add_node(Flatten(), name='flatten-%s' % fsz, input='maxpool-%s' % fsz)

if len(filter_sizes)>1:
	cnn_graph.add_output(name='output',inputs=['flatten-%s' % fsz for fsz in filter_sizes],merge_mode='concat')
else: 
	cnn_graph.add_output(name='output', input='flatten-%s' % filter_sizes[0])

cnn = Sequential()
cnn.add(Embedding(len(cnn_vocabulary), 300, input_length=32,weights=cnn_embedding_weights))
cnn.add(Dropout(0.25, input_shape=(32, 300)))
cnn.add(cnn_graph)

lstm = Sequential()
lstm.add(Embedding(len(lstm_vocabulary), 300, input_length=667,weights=lstm_embedding_weights))
lstm.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))

model = Sequential()
model.add(keras.layers.core.Merge([cnn, lstm], mode='concat'))
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Dense(27))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#skf = StratifiedKFold(Y_train, n_folds=10,shuffle=True)
#for train_index, test_index in skf:
#	model.fit([cnn_train[train_index], lstm_train[train_index]], Y_train_f[train_index], batch_size=128, nb_epoch=20,show_accuracy=True)
#	print(model.test_on_batch([cnn_train[test_index], lstm_train[test_index]], Y_train_f[test_index], accuracy=True))

model.fit([cnn_shuffled, lstm_shuffled], Y_train, batch_size=128, nb_epoch=20,show_accuracy=True,validation_split=0.3)
#print(test_on_batch(X, y, accuracy=False, sample_weight=None))