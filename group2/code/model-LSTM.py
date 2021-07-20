# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:19:21 2021

@author: rania
"""
import os 
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.models import Sequential


# Loading preprossesed Data

print('Loading data...')
x_train = pickle.load(open("wordcount.pkl","rb"))
print(x_train.shape)
y_train = pickle.load(open("label.pkl","rb"))


print('Build model...')

# Model Constants
max_words= 5000
max_phrase_len= x_train.shape[1]
batch_size = 32
embedding_dims = 50
epochs = 10

#Build the model 
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim = max_phrase_len, output_dim = 64))
model_lstm.add(LSTM(64, dropout = 0.1))
model_lstm.add(Dense(256))
model_lstm.add(Dense(7, activation = 'relu'))
model_lstm.add(Dense(1))

#Custom Loss
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    
#Compile the model
model_lstm.compile(
    loss=root_mean_squared_error,
    optimizer='adam',
    metrics=['accuracy']
)


print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=max_phrase_len)
print('x_train shape:', x_train.shape)

# Checkpoint saving
checkpoint_path = "training_1\cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback =ModelCheckpoint(checkpoint_path, monitor = 'accuracy',save_best_only=True,save_weights_only=True,verbose=1)

print('Train...')
# training 
model_lstm.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[cp_callback])

print('fin')

#saving the model
model_lstm.save("network.h5")