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
from terminaltables import AsciiTable

parser = argparse.ArgumentParser(\
        prog='Train a feedforward neural network for the forward dynamics of a drone',\
        description=''
        )


parser.add_argument('-train_dataset_path', default='', help='path to training dataset')
parser.add_argument('-val_dataset_path', default='', help='path to validation dataset')
parser.add_argument('-epochs', default=10, help='number of neurons within hidden layers')
parser.add_argument('-model_name', default='forward_dynamics_model', help='name of model')
parser.add_argument('-model_path', default='./trained_models', help='path to direcory to save model')
parser.add_argument('-lr', default=0.003, help='learning rate')
parser.add_argument('-w_reg', default=0.003, help='weight regularistion')
parser.add_argument('-Nt', default=10, help='window size')
parser.add_argument('-batch', default=32, help='batch size')


args = parser.parse_args()

dataset_path = str(vars(args)['train_dataset_path'])
epochs = int(vars(args)['epochs'])
name_of_model = str(vars(args)['model_name'])
path_to_model = str(vars(args)['model_path'])
lr = float(vars(args)['lr'])
weight_regularisation = float(vars(args)['w_reg'])
window_size = int(vars(args)['Nt'])
batch_size = int(vars(args)['batch'])


dataset_readme = dataset_path+'_readme'


print('----------------------------------------------------------------')
print('Fetching training info from: {}'.format(dataset_readme))
print('----------------------------------------------------------------')
data = []
with shelve.open(dataset_readme) as db:
    for key,value in db.items():
        data.append([str(key),str(value)])
db.close()
table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)

# num_samples = data['dataset_num_entries']
print(data)


print('\n--------------------------------------------------------------')
print('Saving dataset readme at:', str(path_to_model + '/'+ name_of_model +'_readme'))
print('--------------------------------------------------------------')

with shelve.open( str(path_to_model + '/'+ name_of_model + '_readme')) as db:


    with shelve.open(dataset_readme) as db2:
        for key,value in db2.items():
            db[str(key)] = value
    db2.close()

    db['epochs'] = epochs
    db['learning_rate'] = lr
    db['weight_regularisation'] = weight_regularisation
    db['window_size'] = window_size


db.close()

#  Loss
mae = tf.keras.losses.MeanAbsoluteError()
# Weight regularistion

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# Building model
def create_ffnn_model(input_shape=10):
    model = keras.Sequential([
    layers.Dense(200,input_shape=(input_shape,), dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(200,input_shape=(input_shape,), dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(6)])
    return model

def create_lstm_model():
    pass


# Needs some cleaning
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        # print(dataset[:,indices].flatten())
        data.append(dataset[:,indices].flatten())

        if single_step:
            labels.append(dataset[:,i+target_size].flatten())
            # print(labels)
            # labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

# Load dataset
def load_dataset(path_to_h5py):

    print('\n--------------------------------------------------------------')
    print('Reading dataset file: {}'.format(path_to_h5py))
    print('--------------------------------------------------------------')
    f = h5py.File(path_to_h5py, 'r')
    # print('{} contains: {}'.format(path_to_h5py,f.keys()))
    dataset = f['dataset']
    return dataset


# Needs some cleaning
def load_bad_tf_dataset(path2h5py,windowsize,train_split):

    print('\n--------------------------------------------------------------')
    print('Reading dataset file: {}'.format(path_to_h5py))
    print('--------------------------------------------------------------')
    f = h5py.File(path_to_h5py, 'r')
    # print('{} contains: {}'.format(path_to_h5py,f.keys()))
    dataset = f['dataset']


    past_history = windowsize # Time window
    future_target = 0  # How far in the future does the nn predict
    STEP = 1 # Step between samples
    TRAIN_SPLIT=train_split #

    x_train, y_train = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    return train_data_single

def load_eff_tf_dataset():
    pass




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


train_dataset = load_dataset(dataset_path)
forward_dynamics_model = create_model()
#
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
# counter = 0



for epoch in range(epochs):

    counter = 0
#
    for (x_train, y_train) in train_dataset:
        train_step(model, optimizer, x_train, y_train)
#
#     with train_summary_writer.as_default():
#         tf.summary.scalar('loss', train_loss.result(), step=epoch)
#         tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
#
#     for (x_test, y_test) in test_dataset:
#         test_step(model, x_test, y_test)
#
#     with test_summary_writer.as_default():
#         tf.summary.scalar('loss', test_loss.result(), step=epoch)
#         tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print (template.format(epoch+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          test_loss.result(),
#                          test_accuracy.result()*100))
