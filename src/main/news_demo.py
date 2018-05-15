from keras.layers          import Lambda, Input, Dense, GRU, LSTM, Dropout
from keras.models          import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import Adam
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import random
import sys
import pickle
import glob
import copy
import os
import re
import MeCab
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from gensim.models import word2vec
import numpy as np
from keras.utils import plot_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import initializers
from sklearn.decomposition import PCA
import json
from keras.models import load_model

word2vecModel = word2vec.Word2Vec.load(
    '/mnt/sdc/wikipedia_data/jawiki_wakati.model')
addDict = {}
seq_len = 13
categories = 6


def predictVector(word, around_words_list):
    global addDict
    if word in addDict:
        vector = addDict[word]
    else:
        addUnknownWord(word, around_words_list)
        vector = addDict[word]
    return vector


def addUnknownWord(word, around_words_list):
    rand_vector = np.random.rand(
        200) / np.linalg.norm(np.random.rand(200)) * (10 + 3 * np.random.rand(1))
    vector = np.array(word2vecModel[word2vecModel.predict_output_word(
        around_words_list)[0][0]]) + rand_vector
    addDict[word] = vector


def Wakati(text):
    m = MeCab.Tagger(
        "-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd -Owakati")
    result = m.parse(text)
    ws = re.compile(" ")
    words = [word for word in ws.split(result)]
    if words[-1] == u"\n":
        words = words[:-1]
    return words


def seq2vecs(words):
    vectors = []
    for i in range(len(words)):
        try:
            vector = word2vecModel[words[i]]
            vectors.append(vector)
        except:
            try:
                vectors.append(predictVector(words[i], [words[i - 1]]))
            except:
                if i != 0:
                    return []
                else:
                    similar_word = word2vecModel.similar_by_vector(
                        addDict[words[i - 1]], topn=10, restrict_vocab=None)[0][0]
                    vectors.append(predictVector(words[i], [similar_word]))
    return vectors

class AttLayer(Layer):
    
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = K.variable(self.init((input_shape[-1],1)))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x,self.W))
        eij=K.squeeze(eij,axis=2)
        ai = K.exp(eij)
        Sum=K.expand_dims(K.sum(ai, axis=1),axis=1)
        weights = ai/Sum
        weights=K.expand_dims(weights,axis=1)
        return weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

def encode(args):
    x,weights=args
    weighted_input = K.batch_dot(weights, x)
    weighted_input=K.squeeze(weighted_input,axis=1)
    return weighted_input

category = ["国際", "経済", "エンタメ", "スポーツ", "IT", "科学"]
model = load_model('../result/lstm_attention.h5', {"AttLayer": AttLayer})

while True:
    try:
        title = input()
    except:
        break
    words = Wakati(title.strip())
    vecs = seq2vecs(words)
    if len(vecs) == 0:
        print("ベクトルに変換できませんでした")
        continue
    elif len(vecs) > seq_len:
        vecs = vecs[:seq_len]
    x = [[0.] * 200 for _ in range(seq_len)]
    x[0:len(vecs)] = vecs
    y = [x]
    print(category[model.predict(np.array(y))[0].argmax()])

