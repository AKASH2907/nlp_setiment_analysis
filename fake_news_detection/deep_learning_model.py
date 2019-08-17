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

datapath = './dataset/'

# Reading datasets

train = pd.read_csv(datapath + 'train.tsv', header=None, index_col=0, delimiter='\t')
val = pd.read_csv(datapath + 'val.tsv', header=None, index_col=0, delimiter='\t')
test = pd.read_csv(datapath + 'test.tsv', header=None, index_col=0, delimiter='\t')

# train.head()

# Assign Column variables

train.columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'job title', 
                 'state info', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context', 
                 'justification']

val.columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'job title', 
                 'state info', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context', 
                 'justification']

test.columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'job title', 
                 'state info', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context', 
                 'justification']

# Dropping non useful columns

train.drop(['ID','job title', 'state info'], axis=1, inplace=True)
val.drop(['ID','job title', 'state info'], axis=1, inplace=True)
test.drop(['ID','job title', 'state info'], axis=1, inplace=True)
train.drop([2142, 9375], axis=0, inplace=True)

# Verify the number of null values and filling no_text accordingly

train['justification'] = train['justification'].fillna("no_text")
val['justification'] = val['justification'].fillna("no_text")
test['justification'] = test['justification'].fillna("no_text")

print(train.shape)
print(val.shape)
print(test.shape)


#  Text preprocessing with regex library

import re

all_data = pd.concat([train, val, test], ignore_index=False)
# print(all_data.head())
print(all_data.shape)

all_statements = []
all_justifications = []


for i in all_data['statement']:
    all_statements.append(i)

for line in all_data['justification']:
    all_justifications.append(line)

print(len(all_statements))
print(len(all_justifications))

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

statements_clean = preprocess_reviews(all_statements)
justifications_clean = preprocess_reviews(all_justifications)


print(all_data.label.unique())

# Target variables

Y_binary = []
# all_data.label.unique().tolist()
for i in all_data['label']:
    if i=='false' or i=='barely-true' or i=='pants-fire':
        Y_binary+=[0]
    else:
        Y_binary+=[1]

Y_six = []

for i in all_data['label']:
    if i =='pants-fire':
        Y_six+=[0]
    elif i=='false':
        Y_six+=[1]
    elif i=='barely-true':
        Y_six+=[2]
    elif i=='half-true':
        Y_six+=[3]
    elif i=='mostly-true':
        Y_six+=[4]
    else:
        Y_six+=[5]
        
# train.label.unique()
print(len(Y_binary), len(Y_six))

# Separating X_train, X_val & X_test

X_train = statements_clean[:train.shape[0]]
X_val = statements_clean[train.shape[0]:train.shape[0]+val.shape[0]]
X_test = statements_clean[train.shape[0]+val.shape[0]:]
print(len(X_train), len(X_val), len(X_test))

X_trainj = justifications_clean[:train.shape[0]]
X_valj = justifications_clean[train.shape[0]:train.shape[0]+val.shape[0]]
X_testj = justifications_clean[train.shape[0]+val.shape[0]:]

Y_train_bi = Y_binary[:train.shape[0]]
Y_val_bi = Y_binary[train.shape[0]:train.shape[0]+val.shape[0]]
Y_test_bi = Y_binary[train.shape[0]+val.shape[0]:]

Y_train_six = Y_six[:train.shape[0]]
Y_val_six = Y_six[train.shape[0]:train.shape[0]+val.shape[0]]
Y_test_six = Y_six[train.shape[0]+val.shape[0]:]

Y_train_six = np.asarray(Y_train_six)
Y_val_six = np.asarray(Y_val_six)
Y_test_six = np.asarray(Y_test_six)

Y_train_six = np_utils.to_categorical(Y_train_six, 6)
Y_val_six = np_utils.to_categorical(Y_val_six, 6)
Y_test_six = np_utils.to_categorical(Y_test_six, 6)
print(Y_train_six.shape, Y_val_six.shape, Y_test_six.shape)

# Word TOkenizer Statements

freq_words_state=4000

tokenizer_state = Tokenizer(num_words=freq_words_state)
tokenizer_state.fit_on_texts(X_train)

X_train = tokenizer_state.texts_to_sequences(X_train)
X_val = tokenizer_state.texts_to_sequences(X_val)
X_test = tokenizer_state.texts_to_sequences(X_test)


# Word Tokenizer Justification

freq_words_justify=6000

tokenizer_justify = Tokenizer(num_words=freq_words_justify)
tokenizer_justify.fit_on_texts(X_trainj)

X_trainj = tokenizer_justify.texts_to_sequences(X_trainj)
X_valj = tokenizer_justify.texts_to_sequences(X_valj)
X_testj = tokenizer_justify.texts_to_sequences(X_testj)

vocab_size_state = len(tokenizer_state.word_index) + 1
print(vocab_size_state)
maxlen_s = 100

X_train = pad_sequences(X_train, maxlen=maxlen_s, padding='post')
X_val = pad_sequences(X_val, maxlen=maxlen_s, padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen_s, padding='post')

vocab_size_justify = len(tokenizer_justify.word_index) + 1
print(vocab_size_justify)
maxlen_j = 150

X_trainj = pad_sequences(X_trainj, maxlen=maxlen_j, padding='post')
X_valj = pad_sequences(X_valj, maxlen=maxlen_j, padding='post')
X_testj = pad_sequences(X_testj, maxlen=maxlen_j, padding='post')

# Importing GloVe embeddings

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix_s = zeros((vocab_size_state, 100))
for word, index in tokenizer_state.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_s[index] = embedding_vector
    
print(embedding_matrix_s.shape)

embedding_matrix_j = zeros((vocab_size_justify, 100))
for word, index in tokenizer_justify.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_j[index] = embedding_vector
    
print(embedding_matrix_j.shape)

# Model

embedding_layer_s = Embedding(vocab_size_state, 100, weights=[embedding_matrix_s],
                              input_length=maxlen_s , trainable=False)
print(embedding_layer_s)

embedding_layer_j = Embedding(vocab_size_justify, 100, weights=[embedding_matrix_j],
                              input_length=maxlen_j , trainable=False)
print(embedding_layer_j)

seq_input1 = Input(shape=(maxlen_s, ), dtype='int32')
seq_input2 = Input(shape=(maxlen_j, ), dtype='int32')

emb_s = embedding_layer_s(seq_input1)
emb_j = embedding_layer_j(seq_input2)

gru11 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(emb_s)
gru12 = Bidirectional(GRU(64, return_sequences=False), merge_mode='concat')(gru11)

gru21 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(emb_j)
gru22 = Bidirectional(GRU(64, return_sequences=False), merge_mode='concat')(gru21)

merge1 = Concatenate(axis=-1)([gru12, gru22])

dense1 = Dense(6, activation='softmax')(merge1)

model = Model(inputs=[seq_input1, seq_input2], outputs=dense1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())

# Training model

history = model.fit([X_train, X_trainj], Y_train_six, batch_size=128, epochs=5, verbose=1,
             validation_data=([X_val, X_valj], Y_val_six))

# Print Loss and accuracy

loss, acc = model.evaluate([X_val, X_valj], Y_val_six, verbose=1)
print(acc)

loss, acc = model.evaluate([X_test, X_testj], Y_test_six, verbose=1)
print(acc)
