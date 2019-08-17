import numpy as np
import pandas as pd
import re

from numpy import array
from numpy import asarray
from numpy import zeros

import nltk
import nltk.corpus
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Algorithms
from keras.layers import Flatten,LSTM,Conv1D,Dropout,Bidirectional, GRU, RNN, Concatenate
from keras.layers import SpatialDropout1D
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
from keras.optimizers import SGD, Adam

glove_file = open('glove.6B.50d.txt', encoding="utf8")

embeddings_dictionary = dict()


for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# nltk.download('stopwords')
# nltk.download('punkt')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

topics = train.topic.unique().tolist()

l = list(range(21))
labels = {i : topics[i] for i in range(0, len(topics))}
d = dict(zip(topics, l))


train['labels'] = train["topic"].apply(lambda x: d[x])
all_data = pd.concat([train, test], ignore_index=False)

all_data['Review Text'] = all_data['Review Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
all_data['Review Text'] = all_data['Review Text'].str.replace('[^\w\s]','')

# stop = stopwords.words('english')
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
all_data['Review Text'] = all_data['Review Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


freq = pd.Series(' '.join(all_data['Review Text']).split()).value_counts()[:10]
freq = list(freq.index)
all_data['Review Text'] = all_data['Review Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


freq = pd.Series(' '.join(all_data['Review Text']).split()).value_counts()[-30:]
freq = list(freq.index)
all_data['Review Text'] = all_data['Review Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

st = SnowballStemmer('english')
all_data['Review Text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# from textblob import Word
# all_data['Review Text'] = all_data['Review Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# titles = []
# reviews= []

# for line in all_data['Review Title']:
#     titles.append(line)

# for line in all_data['Review Text']:
#     reviews.append(line)

# print(len(titles))
# print(len(reviews))

# REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
# REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# NO_SPACE = ""
# SPACE = " "

# def preprocess_reviews(reviews):
    
#     reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
#     reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
#     return reviews

# titles_clean = preprocess_reviews(titles)
# reviews_clean = preprocess_reviews(reviews)


# X_train = titles_clean[:train.shape[0]]
# X_test = titles_clean[train.shape[0]:]

# print(len(X_train), len(X_test))

# X_trainj = reviews_clean[:train.shape[0]]
# X_testj = reviews_clean[train.shape[0]:]

X_trainj = all_data['Review Text'][:train.shape[0]]
X_testj = all_data['Review Text'][train.shape[0]:]

Y = []
for i in train['labels']:
    Y+=[i]

Y = np.asarray(Y)
Y = np_utils.to_categorical(Y, 21)

# freq_words_state=400

# tokenizer_state = Tokenizer(num_words=freq_words_state)
# tokenizer_state.fit_on_texts(X_train)


# X_train = tokenizer_state.texts_to_sequences(X_train)
# # X_val = tokenizer_state.texts_to_sequences(X_val)
# X_test = tokenizer_state.texts_to_sequences(X_test)


# vocab_size_state = len(tokenizer_state.word_index) + 1
# print(vocab_size_state)
# maxlen_s = 30

# X_train = pad_sequences(X_train, maxlen=maxlen_s, padding='post')
# # X_val = pad_sequences(X_val, maxlen=maxlen_s, padding='post')
# X_test = pad_sequences(X_test, maxlen=maxlen_s, padding='post')



# embedding_matrix_s = zeros((vocab_size_state, 50))
# for word, index in tokenizer_state.word_index.items():
#     embedding_vector = embeddings_dictionary.get(word)
#     if embedding_vector is not None:
#         embedding_matrix_s[index] = embedding_vector
    
# print(embedding_matrix_s.shape)

# # Output dim of embedding layer according to the glove dimension vector

# embedding_layer_s = Embedding(vocab_size_state, 50, weights=[embedding_matrix_s],
#                               input_length=maxlen_s , trainable=False)
# print(embedding_layer_s)


freq_words_justify=2000

tokenizer_justify = Tokenizer(num_words=freq_words_justify)
tokenizer_justify.fit_on_texts(X_trainj)

X_trainj = tokenizer_justify.texts_to_sequences(X_trainj)
# X_valj = tokenizer_justify.texts_to_sequences(X_valj)
X_testj = tokenizer_justify.texts_to_sequences(X_testj)
print(len(X_trainj))

vocab_size_justify = len(tokenizer_justify.word_index) + 1
print(vocab_size_justify)
maxlen_j = 200

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

seq_input = Input(shape=(maxlen_j, ), dtype='int32')
x = embedding_layer_j(seq_input)
x = SpatialDropout1D(0.2)(x)  
x = Bidirectional(LSTM(128))(x)  
# x = Conv1D(128, 5, activation='relu')(x)
# x = Dropout(0.3)(x)
# x = Conv1D(256, 5, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
x = Dense(21, activation='softmax')(x)

model = Model(inputs=seq_input, outputs=x)
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4/25)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_trainj, Y, batch_size=128, epochs=25, verbose=1)

loss, acc= model.evaluate(X_trainj, Y, verbose=1)
print(acc)
t = model.predict(X_testj)

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

# submission.to_csv('submit_dl6.csv', index=False)