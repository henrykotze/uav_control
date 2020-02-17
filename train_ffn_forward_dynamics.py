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
import os

parser = argparse.ArgumentParser(\
        prog='Train a feedforward neural network for the forward dynamics of a drone',\
        description=''
        )


parser.add_argument('-train_dataset_path', default='', help='path to training dataset')
parser.add_argument('-epochs', default=10, help='epochs for training')
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





# SAVING METRIC
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
completed_log_dir = './training_results/' + current_time
train_log_dir = './training_results/'+ current_time + '/train'
test_log_dir = './training_results/' + current_time + '/validation'
checkpoint_log_dir = './training_results/' + current_time + '/checkpoints'
checkpoint_log_path = './training_results/' + current_time + '/checkpoints/cp-epoch-{}.h5'
os.mkdir(completed_log_dir)
os.mkdir(checkpoint_log_dir)



print('\n--------------------------------------------------------------')
print('Saving model training readme at:', str(completed_log_dir+ '/'+ 'readme'))
print('--------------------------------------------------------------')

with shelve.open( str(completed_log_dir + '/'+ 'readme')) as db:

    with shelve.open(dataset_readme) as db2:
        name_validation_dataset = db2['name_of_validation_dataset']
        dataset_loc = db2['dataset_loc']
        for key,value in db2.items():
            db[str(key)] = value


    db2.close()

    db['epochs'] = epochs
    db['learning_rate'] = lr
    db['weight_regularisation'] = weight_regularisation
    db['window_size'] = window_size
    db['batch_size'] = batch_size


db.close()


with shelve.open( str(checkpoint_log_dir + '/'+ 'readme')) as db:

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


train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# CUSTOM TRAINING
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()

#  METRICS
mean_train_loss = tf.keras.metrics.Mean(name='mean_train_loss')
mean_val_loss = tf.keras.metrics.Mean(name='mean_val_loss')
prev_mean_val_loss = 10000;

#


train_mean_sqrd_error = tf.keras.metrics.MeanSquaredError(name='train_mean_sqrd_error')
val_mean_sqrd_error = tf.keras.metrics.MeanSquaredError(name='val_mean_sqrd_error')
train_mean_abs_error = tf.keras.metrics.MeanAbsoluteError(name='train_mean_abs_error')
val_mean_abs_error = tf.keras.metrics.MeanAbsoluteError(name='val_mean_abs_error')

# Building model
def create_ffnn_model(input_shape=10):

    model = keras.Sequential([
    layers.Dense(100,input_shape=(input_shape,)), \
    layers.ReLU(),\
    layers.Dense(100,dtype=tf.float64),\
    layers.ReLU(),\
    layers.Dense(6,dtype=tf.float64)
    ])
    model.summary()

    return model



# Building model
def create_lstm_model(batchsize,timesteps,features):


    model = keras.Sequential([
    # keras.layers.TimeDistributed(keras.layers.Dense(8),input_shape=(timesteps,features) ),
    keras.layers.LSTM(10,input_shape=(batchsize,timesteps,features)),
    keras.layers.Dense(10),
    keras.layers.ReLU()
    ])

    model.summary()
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
        mean_train_loss.update_state(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test_step(model, x_test, y_test):
    predictions = model.predict_on_batch(x_test)

    loss = mae_and_weight_reg_loss(predictions,y_test,model.trainable_variables)
    mean_val_loss.update_state(loss)



validation_dataset_path = str(dataset_loc) + str(name_validation_dataset)

# [q1,q2,q3,q4,U,V,W,T1,T2,T3,T4]
input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [11, 12, 13, 14, 15, 16]
train_dataset = esl_timeseries_dataset(dataset_path,window_size,step,batch_size,
                                        input_indices,output_indices)

validation_dataset = esl_timeseries_dataset(validation_dataset_path,window_size,step,batch_size,
                                        input_indices,output_indices)

# def create_lstm_model(batchsize,timesteps,features):

forward_dynamics_model = create_lstm_model(batch_size,window_size,11)
keras.utils.plot_model(forward_dynamics_model, str(completed_log_dir + '/'+ name_of_model + '.png'), show_shapes=True)

# for epoch in trange(epochs):
for epoch in range(epochs):

    train_progressbar = trange(train_dataset.getTotalBatches(), desc='training batch #', leave=True)
    test_progressbar = trange(validation_dataset.getTotalBatches(), desc='validation batch #', leave=True)

    for x_train, y_train in train_dataset:
        train_progressbar.set_description("training batch")
        train_progressbar.refresh() # to show immediately the update
        train_progressbar.update()

        train_step(forward_dynamics_model, optimizer, x_train, y_train)

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', mean_train_loss.result(), step=epoch)
        tf.summary.scalar('train_mean_sqrd_error', train_mean_sqrd_error.result(), step=epoch)
        tf.summary.scalar('train_mean_abs_error', train_mean_abs_error.result(), step=epoch)

    for (x_test, y_test) in validation_dataset:

        test_progressbar.set_description("validation batch")
        test_progressbar.refresh() # to show immediately the update
        test_progressbar.update()

        test_step(forward_dynamics_model, x_test, y_test)

    print("")

    if(mean_val_loss.result() < prev_mean_val_loss):

        print('val loss improved from {} to {}'.format(prev_mean_val_loss,mean_val_loss.result()))
        prev_mean_val_loss = mean_val_loss.result()
        print(checkpoint_log_path.format(epoch))
        forward_dynamics_model.save(checkpoint_log_path.format(epoch))


        with shelve.open( str(checkpoint_log_dir + '/'+ 'readme')) as db:

            db['train_mean_abs_error'] = float(train_mean_abs_error.result())
            db['mean_train_loss'] = float(mean_train_loss.result())
            db['mean_val_loss'] = float(mean_val_loss.result())

        db.close()


    with test_summary_writer.as_default():
        tf.summary.scalar('loss', mean_val_loss.result(), step=epoch)

    print("Epoch {}, train loss: {}, train mae: {}".format(epoch+1,mean_train_loss.result(),train_mean_abs_error.result()))

forward_dynamics_model.save(str(completed_log_dir + '/'+ name_of_model + '.h5'))

with shelve.open( str(completed_log_dir + '/'+ 'readme')) as db:

    db['train_mean_abs_error'] = float(train_mean_abs_error.result())
    db['mean_train_loss'] = float(mean_train_loss.result())
    db['train_log_dir'] = str(train_log_dir)
    db['test_log_dir'] = str(test_log_dir)
    db['mean_val_loss'] = float(mean_val_loss.result())

db.close()
