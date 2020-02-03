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
from tqdm import tnrange, trange
import datetime

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
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print('----------------------------------------------------------------')


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


# SAVING METRIC
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './training_results/'+ current_time + '/train'
test_log_dir = './training_results/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# CUSTOM TRAINING
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()

#  METRICS
train_loss = 0
val_loss = 0


train_mean_sqrd_error = tf.keras.metrics.MeanSquaredError(name='train_mean_sqrd_error')
val_mean_sqrd_error = tf.keras.metrics.MeanSquaredError(name='val_mean_sqrd_error')
train_mean_abs_error = tf.keras.metrics.MeanAbsoluteError(name='train_mean_abs_error')
val_mean_abs_error = tf.keras.metrics.MeanAbsoluteError(name='val_mean_abs_error')

# Building model
def create_ffnn_model(input_shape=10):

    model = keras.Sequential([
    layers.Dense(100,input_shape=(input_shape,),dtype=tf.float64), \
    layers.ReLU(),\
    layers.Dense(6,dtype=tf.float64)
    ])

    return model



def mae_and_weight_reg_loss(predictions,ground_truth,vars):
    loss1 = mae(predictions,ground_truth)
    loss2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])*weight_regularisation
    return loss1+loss2


def train_step(model, optimizer, x_train, y_train):

  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    train_mean_sqrd_error.update_state(y_train,predictions)
    train_mean_abs_error.update_state(y_train,predictions)
    loss = mae_and_weight_reg_loss(y_train, predictions, model.trainable_variables)
    train_loss = loss

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  # train_mean(loss)

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    val_loss = loss



# [q1,q2,q3,q4,U,V,W,T1,T2,T3,T4]
input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices= [11, 12, 13, 14, 15, 16]
train_dataset = esl_timeseries_dataset(dataset_path,window_size,step,batch_size,
                                        input_indices,output_indices)


forward_dynamics_model = create_ffnn_model(train_dataset.get_input_shape())

# for epoch in trange(epochs):
for epoch in range(epochs):
    for x_train, y_train in train_dataset:
        train_step(forward_dynamics_model, optimizer, x_train, y_train)
# #
    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss, step=epoch)
        tf.summary.scalar('train_mean_sqrd_error', train_mean_sqrd_error.result(), step=epoch)
        tf.summary.scalar('train_mean_abs_error', train_mean_abs_error.result(), step=epoch)

# #
# #     for (x_test, y_test) in test_dataset:
# #         test_step(model, x_test, y_test)
# #
# #     with test_summary_writer.as_default():
# #         tf.summary.scalar('loss', test_loss.result(), step=epoch)
# #
    print("Epoch {}, mae: {}".format(epoch+1,train_mean_abs_error.result()))



with shelve.open( str(path_to_model + '/'+ name_of_model + '_readme')) as db:

    db['mean_of_model'] = train_mean_abs_error.result()
    db['train_log_dir'] = str(train_log_dir)
    db['test_log_dir'] = str(test_log_dir)

db.close()
