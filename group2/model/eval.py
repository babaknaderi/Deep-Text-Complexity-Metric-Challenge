# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:31:38 2021

@author: rania
"""
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import sys
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import tensorflow_hub as hub

#'C:/Users/rania/Desktop/group2/testing.csv'
#load testing data
path= sys.argv[-1]
testing = pd.read_csv(path)
sentence_test = testing.sentence 

#constante
loss= 'mse'
metrics = tf.keras.metrics.RootMeanSquaredError()
epochs = 5
init_lr = 3e-5
steps_per_epoch = 1
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
#define optimizer
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

#load the model
path = "Model_Bert_en_uncased_L-8_H-512_A-8" 
model = keras.models.load_model(path)

#compile the model
model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

#predict labels
test_prediction= model.predict(sentence_test)

df = pd.DataFrame()
keys = ["sent_id", "mos"]
df ["sent_id"]= testing.sent_id
df["mos"] = test_prediction[:]
print(df)

#save the result
df.to_csv("Result.csv", sep=",",index=False)

