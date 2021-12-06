import pandas as pd
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
import streamlit as st
#rubric: scikit-learn/Keras
#https://www.kaggle.com/bayunova/is-this-sentence-completed
df = pd.read_json("/Users/jiahaoqiu/Documents/UCI 2021 Fall/Math 10/finished_sentences.json")
#create a set of punctuation as filter
exclude = set(string.punctuation)
def rem_punc(text):
    text = ''.join(ch for ch in text if ch not in exclude)
    return text
#remove punctuation and get the max length of sentence 
df["sentence_clean"] = df["sentence"].apply(lambda x: rem_punc(x))
df["is_finished_num"] = [1 if x == 'Finished' else 0 for x in df['is_finished']]
df["sentence_length"] = df["sentence_clean"].apply(lambda x: len(x))
max_length = df["sentence_length"].max()

def string_to_num(s):
    s_len = len(s)
    sent = s
    tk = tf.keras.preprocessing.text.Tokenizer(lower = True)
    tk.fit_on_texts(sent)
    s_seq = tk.texts_to_sequences(sent)
    s_pad = tf.keras.preprocessing.sequence.pad_sequences(s_seq, maxlen = s_len, padding = 'post')
    return s_pad

# Transform string to numerical value Using tokenizer from tensorflow
X, y = (df['sentence'].values, df['is_finished_num'].values)
tk = tf.keras.preprocessing.text.Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen = 100, padding = 'post')
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.2)

#modeling
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (100,)),
        #keras.layers.Flatten(),
        keras.layers.Dense(4, activation="sigmoid"),
        keras.layers.Dense(4, activation="sigmoid"),
        keras.layers.Dense(2, activation="sigmoid")
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)
history = model.fit(X_train,y_train,epochs=10,validation_split = 0.2)

#Showing performance of the model after different epochs
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
#s = string_to_num()
predictions = model.predict(X_test).argmax(axis =1)
category = {0:"Unfinished", 1:"finished"}
pred = [x for x in predictions] 
actu = [x for x in y_test]
df1 = pd.DataFrame({"Prediction":pred, "Actual":actu})
df1['correctness'] = (df1['Prediction'] == df1['Actual'])
df1
st.title("Over-fitting")