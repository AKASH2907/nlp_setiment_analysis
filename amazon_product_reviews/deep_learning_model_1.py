import numpy as np
import pandas as pd
import re


# Algorithms
from keras.layers import Flatten,LSTM,Conv1D,Dropout,Bidirectional, GRU, RNN, Concatenate
from keras.utils import plot_model, np_utils
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten, Conv1D, RNN, LSTM, GlobalAveragePooling1D, Dense
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

topics = train.topic.unique().tolist()

l = list(range(21))
labels = {i : topics[i] for i in range(0, len(topics))}
d = dict(zip(topics, l))
# print(labels)
# print(d)

train['labels'] = train["topic"].apply(lambda x: d[x])
all_data = pd.concat([train, test], ignore_index=False)
titles = []
reviews= []

for i in all_data['Review Title']:
    titles.append(i)

for line in all_data['Review Text']:
    reviews.append(line)

print(len(titles))
print(len(reviews))

import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

titles_clean = preprocess_reviews(titles)
reviews_clean = preprocess_reviews(reviews)


X_train = titles_clean[:train.shape[0]]
X_test = titles_clean[train.shape[0]:]
print(len(X_train), len(X_test))
X_trainj = reviews_clean[:train.shape[0]]
X_testj = reviews_clean[train.shape[0]:]
Y = []

for i in train['labels']:
    Y+=[i]

Y = np.asarray(Y)
Y = np_utils.to_categorical(Y, 21)
print(Y.shape)

freq_words_state=400

tokenizer_state = Tokenizer(num_words=freq_words_state)
tokenizer_state.fit_on_texts(X_train)


X_train = tokenizer_state.texts_to_sequences(X_train)
# X_val = tokenizer_state.texts_to_sequences(X_val)
X_test = tokenizer_state.texts_to_sequences(X_test)


vocab_size_state = len(tokenizer_state.word_index) + 1
print(vocab_size_state)
maxlen_s = 30

X_train = pad_sequences(X_train, maxlen=maxlen_s, padding='post')
# X_val = pad_sequences(X_val, maxlen=maxlen_s, padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen_s, padding='post')

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.50d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix_s = zeros((vocab_size_state, 50))
for word, index in tokenizer_state.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_s[index] = embedding_vector
    
print(embedding_matrix_s.shape)

# Output dim of embedding layer according to the glove dimension vector

embedding_layer_s = Embedding(vocab_size_state, 50, weights=[embedding_matrix_s],
                              input_length=maxlen_s , trainable=False)
print(embedding_layer_s)


freq_words_justify=2000

tokenizer_justify = Tokenizer(num_words=freq_words_justify)
tokenizer_justify.fit_on_texts(X_trainj)

X_trainj = tokenizer_justify.texts_to_sequences(X_trainj)
# X_valj = tokenizer_justify.texts_to_sequences(X_valj)
X_testj = tokenizer_justify.texts_to_sequences(X_testj)
print(len(X_trainj))

vocab_size_justify = len(tokenizer_justify.word_index) + 1
print(vocab_size_justify)
maxlen_j = 150

X_trainj = pad_sequences(X_trainj, maxlen=maxlen_j, padding='post')
# X_valj = pad_sequences(X_valj, maxlen=maxlen_j, padding='post')
X_testj = pad_sequences(X_testj, maxlen=maxlen_j, padding='post')

embedding_matrix_j = zeros((vocab_size_justify, 50))
for word, index in tokenizer_justify.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_j[index] = embedding_vector
    
print(embedding_matrix_j.shape)

embedding_layer_j = Embedding(vocab_size_justify, 50, weights=[embedding_matrix_j],
                              input_length=maxlen_j , trainable=False)
print(embedding_layer_j)

# seq_input = Input(shape=(maxlen_j, ), dtype='int32')
# x = embedding_layer_j(seq_input)
# x = Conv1D(128, 5, activation='relu')(x)
# x = Conv1D(256, 5, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(21, activation='softmax')(x)

# model = Model(inputs=seq_input, outputs=x)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# print(model.summary())


seq_input1 = Input(shape=(maxlen_s, ), dtype='int32')
seq_input2 = Input(shape=(maxlen_j, ), dtype='int32')

emb_s = embedding_layer_s(seq_input1)
emb_j = embedding_layer_j(seq_input2)

gru11 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(emb_s)
gru12 = Bidirectional(GRU(64, return_sequences=False), merge_mode='concat')(gru11)

gru21 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(emb_j)
gru22 = Bidirectional(GRU(64, return_sequences=False), merge_mode='concat')(gru21)

merge1 = Concatenate(axis=-1)([gru12, gru22])

dense1 = Dense(21, activation='softmax')(merge1)

model = Model(inputs=[seq_input1, seq_input2], outputs=dense1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())


history = model.fit([X_train, X_trainj], Y, batch_size=128, epochs=10, verbose=1)

loss, acc= model.evaluate([X_train, X_trainj], Y, verbose=1)
print(acc)
t = model.predict([X_test, X_testj])

result = []
for i in range(t.shape[0]):
    result+=[np.argmax(t[i])]


final = []
for i in result:
    final.append(labels[i])

submission = pd.DataFrame({
    'Review Text': test['Review Text'],
    'Review Title':test['Review Title'],
    'topic': final
})

submission.to_csv('submit_dl3.csv', index=False)