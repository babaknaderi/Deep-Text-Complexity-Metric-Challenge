# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:51:15 2021

@author: rania
"""
import pandas as pd

# Tensorflow Imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras import backend as K

from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

#Prepare training and test data
#Read data
df = pd.read_excel('training.xlsx')

#split 80-10-10 for training/validation/testing.
train = df[df.split == 0]
validation = df[df.split==1]
test = df[df.split==2]
print('train shape: ',train.shape)
print('test shape: ',test.shape)
x_train=train.Sentence 
y_train=train.MOS
x_val=validation.Sentence 
y_val=validation.MOS
x_test=test.Sentence 
y_test=test.MOS

# Bert Model Constants
batch_size= 32
epochs = 20
init_lr = 3e-5
steps_per_epoch = 1
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)


#Download bert model
bert_model_name = 'small_bert/bert_en_uncased_L-8_H-512_A-8'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
# Load pretrained model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

#Build Model 
def build_model():
  #Input text
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #preprocessing_layer
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  #Load bert model
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  # Add dense layer Output
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

#Custom Loss
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

#Build Model 
model = build_model()

#Def Optimizer
loss= root_mean_squared_error
metrics = tf.keras.metrics.RootMeanSquaredError()
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                         optimizer_type='adamw')

#Compile the model
model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

#Training
print(f'Training model with {tfhub_handle_encoder}')
history = model.fit(x_train,y_train,
                    batch_size=batch_size,
                    validation_data=(x_val,y_val),
                    epochs=epochs)

#testing
loss, RMSE = model.evaluate(x_test,y_test)
print(f'Loss: {loss}')
print(f'RMSE: {RMSE}')

#Plot model training progress
history_dict = history.history
print(history_dict.keys())

RMSE = history_dict['root_mean_squared_error']
loss = history_dict['loss']
val_RMSE = history_dict['val_root_mean_squared_error']
val_loss = history_dict['val_loss']

epochs = range(1, len(RMSE) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training loss')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, RMSE, 'r', label='Training RMSE')
plt.plot(epochs, val_RMSE, 'b', label='val RMSE')
plt.title('Training RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend(loc='lower right')

# saving model
model_save_name = 'Model_Bert_en_uncased_L-8_H-512_A-8'
path = F"/content/gdrive/MyDrive/{model_save_name}" 
model.save(path, include_optimizer=False)