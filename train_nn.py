#!/usr/bin/env python3


# Estimator as a Neural Network for the rotary wing UAV
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import argparse
import pickle
from keras.callbacks import TensorBoard
from datetime import datetime
# from wrap_tensorboard import TrainValTensorBoard
import shelve
import h5py
# import json



parser = argparse.ArgumentParser(\
        prog='Trains the Neural Network ',\
        description=''
        )


parser.add_argument('-dataset', default='./datasets/', help='location of stored dataset, default: ./datasets')
parser.add_argument('-epochs', default=1, help='Number of Epochs, default: 1')
parser.add_argument('-mdl_name', default='nn_mdl', help='Name of model, default: nn_mdl')
parser.add_argument('-reg_w', default='0', help='Regularization of weight, default: 0')
parser.add_argument('-lr', default='0', help='learning rate, default: 0')
parser.add_argument('-valset', default='./datasets/', help='location of validation set, default: ./datasets')
parser.add_argument('-name_dataset', default='dataset0', help='Name Of Training Datasets: ')
parser.add_argument('-name_valset', default='', help='Name Of Validation Datasets: ')
parser.add_argument('-log_loc', default='./training_results/', help='Location of logs and checkpoints')


now = datetime.now()


args = parser.parse_args()

dir = vars(args)['dataset']
epochs = int(vars(args)['epochs'])
mdl_name = str(vars(args)['mdl_name'])
weight_reg = float(vars(args)['reg_w'])
learning_rate = float(vars(args)['lr'])
validation_dir = vars(args)['valset']
training_dataset = vars(args)['name_dataset']
validation_dataset = vars(args)['name_valset']

# Directory containing all different models trained with all their logging
# info
log_dir = vars(args)['log_loc']





# directory with the name of the training session
model_logging_dir = log_dir + '/'+ mdl_name + now.strftime("_%Y%m%d_%H%M%S")
os.mkdir(model_logging_dir)

# location of training dataset
dataset_path = dir + '/' + training_dataset
# location of validation datset
validation_path = validation_dir + '/' + validation_dataset


# Directory containing the logs of the session
tensorboard_logging = model_logging_dir+"/tf_logs"

checkpoint_path = model_logging_dir+"/checkpoints" + '/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
os.mkdir(checkpoint_dir)



with shelve.open( str(model_logging_dir + '/'+'readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)

    with shelve.open( str(dataset_path+'_readme') ) as dataset_readme:
        for key in dataset_readme:
            db[key] = dataset_readme[key]

    dataset_readme.close()
db.close()



print('----------------------------------------------------------------')
print('Training Information: ')
print('----------------------------------------------------------------')
with shelve.open( str(dataset_path+'_readme') ) as db:
    # system = db['system']
    t = int(db['t'])
    numberSims = int(db['numSim'])
    filename = db['filename']

    print("{:<15} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<15} {:<10}".format(key, value))
db.close()

if(validation_dataset != ''):

    print('----------------------------------------------------------------')
    print('Validation Information: ')
    print('----------------------------------------------------------------')
    with shelve.open( str(validation_path+'_readme') ) as db:
        # system = db['system']
        t = int(db['t'])
        numberSims = int(db['numSim'])
        filename = db['filename']

        print("{:<15} {:<10}".format('Label','Value'))
        for key,value in db.items():
            print("{:<15} {:<10}".format(key, value))
    db.close()






tf.reset_default_graph()

# Building model
def build_model(dataset):

    model = keras.Sequential([
    # layers.Dropout(0.4),\
    # layers.Flatten(input_shape=(4,)),\
    layers.Dense(1000,kernel_regularizer=keras.regularizers.l2(weight_reg),input_shape=dataset.output_shapes[0] ), \
    layers.ReLU(),\
    layers.Dense(1000,kernel_regularizer=keras.regularizers.l2(weight_reg),input_shape=dataset.output_shapes[0] ), \
    layers.ReLU(),\
   # layers.Dropout(0.4),\
    # layers.Dense(50,kernel_regularizer=keras.regularizers.l2(weight_reg)),\
    # layers.ReLU(),\
    # layers.Dense(20,kernel_regularizer=keras.regularizers.l2(weight_reg)),\
    # layers.ReLU(),\
    # layers.Dropout(0.2),\
    # layers.Dropout(0.4),\

    layers.Dense(4,kernel_regularizer=keras.regularizers.l2(weight_reg)),\
    # layers.Linear()
    # layers.Softmax()])
    ])

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='mean_absolute_error',    \
                    optimizer=optimizer,        \
                    metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy' ])

    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    # plt.ylim([0,20])
    plt.legend()
    plt.show()





# Getting data
# dir: location of directory containing the *.npz file
# filename: Base filaname to be used
# features: np.array that contains all features
# labels: np.array that contians all labels
def loadData(dir):
    # in the directory, dir, determine how many *.npz files it contains
    with h5py.File(str(dir),'r') as h5f:
        print('==============================================================')
        print('Loading dataset from: ' ,str(dir))
        print('==============================================================\n')
        features = h5f['features'][:]
        labels = h5f['labels'][:]
        h5f.close()


    # each row of `features` corresponds to the same row as `labels`.

    assert features.shape[0] == labels.shape[0]
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    # returns:
    # dataset with correct size and type to match the features and labels
    # features from all files loaded
    # labels from all files loaded
    return [tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)),features,labels]


class saveLogs(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        with shelve.open( str(model_logging_dir + '/'+'readme') ) as db:
            for log_entry in logs:
                db[log_entry] = logs[log_entry]
        db.close()




if __name__ == '__main__':

    # Setting up an empty dataset to load the data into
    [dataset,features,labels] = loadData(dataset_path)

    if(validation_dataset != ''):
        [dataset_val,features_val,labels_val] = loadData(validation_path)



    print('=======================================')
    print('Building Model')
    print('=======================================')
    model = build_model(dataset)

    model.summary()



    # Callback for Checkpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_mean_absolute_error',
                                                 period=1,
                                                 save_best_only=True,
                                                 verbose=1)
    # Callback for TensorBoard
    tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_logging,\
                                            histogram_freq=1,\
                                            write_graph=True,\
                                            write_images=True,\
                                            write_grads=True)



    # Learning of Model

    if(validation_dataset == ''):
        history = model.fit(features, labels, epochs=epochs, \
        validation_split=0.1, verbose=1, callbacks=[saveLogs(),tbCallBack,cp_callback])

    elif(validation_dataset != ''):
        history = model.fit(features, labels, epochs=epochs, \
        validation_data=(features_val,labels_val), verbose=1, callbacks=[saveLogs(),tbCallBack,cp_callback])



    print('\n-----------------------------------')
    print('\n Model Saved at: ' , str(model_logging_dir + '/' + mdl_name))
    print('\n-----------------------------------')
    model.save(model_logging_dir+'/'+mdl_name)
