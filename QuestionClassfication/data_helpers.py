import numpy as np
import re
import itertools
from collections import Counter
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{1,}", " ", string)
    return string.strip().lower()

def cat_map():
    catmap={}
    id=0
    f=open("cat")
    cat=set([s.strip() for s in list(f.readlines())])
    for i in cat:
        catmap[i]=id
        id=id+1
    return catmap

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    cnn_train = list(open("cnn").readlines())
    cnn_train = [s.strip() for s in cnn_train]
    cnn_train = [s.split(" ") for s in cnn_train]
    lstm_train = list(open("lstm").readlines())
    lstm_train = [s.strip() for s in lstm_train]
    lstm_train = [s.split(" ") for s in lstm_train]

    catmap = cat_map()
    Y_train = list(open("cat").readlines())
    Y_train = [s.strip() for s in Y_train]
    Y_train = [catmap[s] for s in Y_train]

    return [cnn_train, lstm_train,Y_train]


def pad_sentences(cnn_train, lstm_train,padding_word="<PAD/>"):
    cnn_sequence_length = max(len(x) for x in cnn_train)
    lstm_sequence_length = max(len(x) for x in lstm_train)
    cnn_padded_sentences = []
    lstm_padded_sentences = []
    for i in range(len(cnn_train)):
        sentence = cnn_train[i]
        cnn_num_padding = cnn_sequence_length - len(sentence)
        cnn_new_sentence = sentence + [padding_word] * cnn_num_padding
        cnn_padded_sentences.append(cnn_new_sentence)
    for i in range(len(lstm_train)):
        sentence = lstm_train[i]
        lstm_num_padding = lstm_sequence_length - len(sentence)
        lstm_new_sentence = sentence + [padding_word] * lstm_num_padding
        lstm_padded_sentences.append(lstm_new_sentence)
    return [cnn_padded_sentences,lstm_padded_sentences]

def build_vocab(cnn_padded_sentences,lstm_padded_sentences):
    cnn_word_counts = Counter(itertools.chain(*cnn_padded_sentences))
    cnn_vocabulary_inv = [x[0] for x in cnn_word_counts.most_common()]
    cnn_vocabulary = {x: i for i, x in enumerate(cnn_vocabulary_inv)}
    lstm_word_counts = Counter(itertools.chain(*lstm_padded_sentences))
    lstm_vocabulary_inv = [x[0] for x in lstm_word_counts.most_common()]
    lstm_vocabulary = {x: i for i, x in enumerate(lstm_vocabulary_inv)}
    return [cnn_vocabulary,cnn_vocabulary_inv,lstm_vocabulary,lstm_vocabulary_inv]


def build_input_data(cnn_padded_sentences, Y_train, cnn_vocabulary,lstm_padded_sentences,lstm_vocabulary):
    cnn_train = np.array([[cnn_vocabulary[word] for word in sentence] for sentence in cnn_padded_sentences])
    lstm_train = np.array([[lstm_vocabulary[word] for word in sentence] for sentence in lstm_padded_sentences])
    Y_train = np.array(Y_train)
    return [cnn_train,lstm_train,Y_train]


def load_data():
    cnn_train,lstm_train,Y_train = load_data_and_labels()
    cnn_padded_sentences,lstm_padded_sentences = pad_sentences(cnn_train,lstm_train)
    cnn_vocabulary,cnn_vocabulary_inv,lstm_vocabulary,lstm_vocabulary_inv = build_vocab(cnn_padded_sentences,lstm_padded_sentences)
    cnn_train,lstm_train,Y_train = build_input_data(cnn_padded_sentences, Y_train, cnn_vocabulary,lstm_padded_sentences,lstm_vocabulary)
    return [cnn_train,lstm_train,Y_train,cnn_vocabulary,cnn_vocabulary_inv,lstm_vocabulary,lstm_vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
