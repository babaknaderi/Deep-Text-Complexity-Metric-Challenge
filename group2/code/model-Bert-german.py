# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:38:04 2021

@author: rania
"""


import pandas as pd
import numpy as np

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from keras import backend as K
import matplotlib.pyplot as plt

#load data
df = pd.read_excel('training.xlsx')
#split data
train = df[df.split == 0]
validation = df[df.split==1]
test = df[df.split==2]
print('train shape: ',train.shape)
print('test shape: ',test.shape)


#load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
MAXLEN = 192

def preprocess_text(data):
    """ take texts and prepare as input features for BERT 
    """
    input_ids = []
    # For every sentence...
    for comment in data:
        encoded_sent = tokenizer.encode_plus(
            text=comment,
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAXLEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            return_attention_mask=False,  # attention mask not needed for our task
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get("input_ids"))
    return input_ids

# preprocess data 
train_ids = preprocess_text(train["Sentence"])
val_ids = preprocess_text(validation["Sentence"])
test_ids = preprocess_text(test["Sentence"])

train_labels = train['MOS']
test_labels = test['MOS']
val_labels=  validation['MOS']
print(f"Train set: {len(train_ids)}\nTest set: {len(test_ids)}")

# model constant
MAXLEN = MAXLEN
BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 1e-5
DATA_LENGTH = len(train)

def create_dataset(
    data_tuple,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    buffer_size=DATA_LENGTH,
    train=False,
):
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    if train:
        dataset = dataset.shuffle(
            buffer_size=buffer_size, reshuffle_each_iteration=True
        ).repeat(epochs)
    dataset = dataset.batch(batch_size)
    return dataset


train = create_dataset(
    (train_ids, train_labels), buffer_size=len(train_ids), train=True
)
test = create_dataset((val_ids, val_labels), buffer_size=len(val_ids))

# loss custom
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

#build the model
def build_model(transformer, max_len=MAXLEN):
    """ add dense layer to pretrained model
    """
    input_word_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids"
    )
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(1, activation=None)(cls_token)
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=out)
    model.compile(
        tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        loss=root_mean_squared_error,
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    return model
# load pretrained bert model
transformer_layers = TFBertModel.from_pretrained("bert-base-german-cased")
# build model
model = build_model(transformer_layers, max_len=MAXLEN)
model.summary()

steps_per_epoch = int(np.floor((len(train_ids) / BATCH_SIZE)))
print(
    f"Model Params:\nbatch_size: {BATCH_SIZE}\nEpochs: {EPOCHS}\n"
    f"Step p. Epoch: {steps_per_epoch}\n"
    f"Learning rate: {LEARNING_RATE}"
)
# train the model
hist = model.fit(
    train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=test,
    verbose=1
)

#eval the model
eval = create_dataset((test_ids, test_labels), buffer_size=len(val_ids))

loss, RMSE = model.evaluate(eval)

pred = model.predict(eval, batch_size=BATCH_SIZE, verbose=2, use_multiprocessing=True)

print(f'Loss: {loss}')
print(f'RMSE: {RMSE}')

#Plot model training progress
history_dict = hist .history
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