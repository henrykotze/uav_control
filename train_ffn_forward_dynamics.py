#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import contrib
import argparse
import sys
import shelve
import h5py
from esl_timeseries_dataset import esl_timeseries_dataset
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
parser.add_argument('-step', default=1, help='step size between samples')


args = parser.parse_args()

dataset_path = str(vars(args)['train_dataset_path'])
epochs = int(vars(args)['epochs'])
name_of_model = str(vars(args)['model_name'])
path_to_model = str(vars(args)['model_path'])
lr = float(vars(args)['lr'])
weight_regularisation = float(vars(args)['w_reg'])
window_size = int(vars(args)['Nt'])
batch_size = int(vars(args)['batch'])
step = int(vars(args)['step'])


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
    db['batch_size'] = batch_size


db.close()

# CUSTOM TRAINING
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(lr)

#  METRICS
train_loss = tf.keras.metrics.MeanAbsoluteError(name='train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.MeanAbsoluteError(name='test_loss', dtype=tf.float32)
train_precision = tf.keras.metrics.Precision(name='train_precision', dtype=tf.float32)
test_precision = tf.keras.metrics.Precisioe(name='test_precision', dtype=tf.float32)

# Building model
def create_ffnn_model(input_shape=10):

    model = keras.Sequential([
    layers.Dense(200,input_shape=(input_shape,), dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(200, dtype='float32',kernel_initializer='glorot_uniform'), \
    layers.ReLU(),\
    layers.Dense(6)])

    return model

def create_lstm_model():
    pass


def mae_and_weight_reg_loss(predictions,ground_truth,vars):
    loss1 = mae(predictions,ground_truth)
    loss2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])*weight_regularisation
    return loss1+loss2


def train_step(model, optimizer, x_train, y_train):

  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = mae_and_weight_reg_loss(y_train, predictions, model.trainable_variables)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)


train_dataset = esl_timeseries_dataset(dataset_path,window_size,step,batch_size)
forward_dynamics_model = create_model()

for epoch in range(epochs):

    for (x_train, y_train) in train_dataset:
        train_step(model, optimizer, x_train, y_train)
#
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
#
#     for (x_test, y_test) in test_dataset:
#         test_step(model, x_test, y_test)
#
#     with test_summary_writer.as_default():
#         tf.summary.scalar('loss', test_loss.result(), step=epoch)
#
    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print(template.format(epoch+1,
                         train_loss.result(),
                         test_loss.result()))




with shelve.open( str(path_to_model + '/'+ name_of_model + '_readme')) as db:

    db['loss_of_training_dataset'] = train_loss.result()
    db['loss_of_validatio_dataset'] = test_loss.result()

db.close()
