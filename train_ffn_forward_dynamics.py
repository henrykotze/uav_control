#!/usr/bin/env python3


import scipy.integrate as spi
import tensorflow as tf
import numpy as np
# from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')
from tensorflow.keras import layers
import pickle
from tensorflow import contrib
import argparse
import sys
import shelve
import argparse
import h5py

parser = argparse.ArgumentParser(\
        prog='Train a feedforward neural network for the forward dynamics of a drone',\
        description=''
        )


parser.add_argument('-train_dataset_path', default='', help='path to training dataset')
parser.add_argument('-val_dataset_path', default='', help='path to validation dataset')
parser.add_argument('-neurons', default=10, help='number of neurons within hidden layers')
parser.add_argument('-epochs', default=10, help='number of neurons within hidden layers')
parser.add_argument('-model_name', default='forward_dynamics_model', help='name of model')
parser.add_argument('-model_path', default='./trained_models', help='path to direcory to save model')


args = parser.parse_args()

model_path = vars(args)['model_path']
num_hidden_layers = vars(args)['layers']
num_neurons = vars(args)['neurons']
epochs = int(vars(args)['epochs'])
name_of_model = str(vars(args)['model_name'])
path_to_model = str(vars(args)['model_path'])



print('----------------------------------------------------------------')
print('')
print('----------------------------------------------------------------')
with shelve.open(model_readme) as db:
    for key,value in db.items():
        print("{}: {}".format(key, value))
db.close()


print('\n--------------------------------------------------------------')
print('Saving dataset readme at:', str(model_path + '/'+ name_of_model +'_readme'))
print('--------------------------------------------------------------')

with shelve.open( str(model_path + '/'+ name_of_model + '_readme')) as db:

    db['epochs'] = epochs
    db['learning_rate'] = lr
    db['weight_regularisation'] = weight_regularisation


db.close()

#  Loss
mae = tf.keras.losses.MeanAbsoluteError()
# Weight regularistion

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# Building model
def create_model(input_shape=25):
    model = keras.Sequential([
    layers.Dense(200,input_shape=(input_shape,), dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(200,input_shape=(input_shape,), dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(6)])
    return model


# Load dataset
def load_dataset(path_to_h5py):
    print('Reading dataset file: {}'.format(path_to_h5py))
    f = h5py.File(path_to_h5py, 'r')
    print('{} contains: {}'.format(path_to_h5py,f.keys())
    dataset = f['dataset']
    return dataset

def train_step(model, optimizer, x_train, y_train):

  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object(y_train, predictions)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  test_loss(loss)
  test_accuracy(y_test, predictions)




forward_dynamics_model = create_model()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()



for epoch in range(epochs)
