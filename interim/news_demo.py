from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.optimizers import Adam
import MeCab
from gensim.models import word2vec
import numpy as np
import re
word2vecModel = word2vec.Word2Vec.load(
    '/mnt/sdc/wikipedia_data/jawiki_wakati.model')
addDict = {}
seq_len = 20
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


inputs = Input(shape=(seq_len, 200))
encoded = LSTM(512, name="encoder")(inputs)
x = Dense(128)(encoded)
decoded = Dense(categories, activation='softmax')(x)
model = Model(inputs=inputs, outputs=decoded)
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=["accuracy"])
model.load_weights('news.hdf5')
category = ["国際", "経済", "エンタメ", "スポーツ", "IT", "科学"]
while True:
    try:
        title = input()
    except:
        break
    words = Wakati(title)
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

