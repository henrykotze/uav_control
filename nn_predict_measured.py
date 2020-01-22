#!/usr/bin/env python3

# Estimator as a Neural Network for the rotary wing UAV
import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import scipy.integrate as spi
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


parser.add_argument('-dataset_name', default='dataset0', help='name of your dataset')
parser.add_argument('-dataset_loc', default='./datasets/', help='location to store dataset')
parser.add_argument('-n', default=1, help='')
parser.add_argument('-model_path', default='./nn_mdl', help='path to neural network model')

args = parser.parse_args()

dir = vars(args)['dataset_loc']
dataset = vars(args)['dataset_name']
dataset_path = dir + '/' + dataset
n = int(vars(args)['n'])
model_path = vars(args)['model_path']

def getReadmePath(path):
    readme = ''
    if 'checkpoints' in path:
        dirs = path.split('/')
        pos = dirs.index("checkpoints")
        for i in range(0,pos):
            readme += dirs[i] + '/'

    else:
        dirs = path.split('/')
        pos = dirs.index("nn_mdl")
        for i in range(0,pos):
            readme += dirs[i] + '/'

    readme += 'readme'
    return readme

model_readme = getReadmePath(model_path)


with shelve.open( str(dataset_path+'_readme') ) as db:
    # system = db['system']
    timeSteps = int(db['timeSteps'])
    Nt  = int(db['Nt'])

    print("{:<15} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<15} {:<10}".format(key, value))
db.close()



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
    return [tf.data.Dataset.from_tensor_slices(features_placeholder),features,labels]





if __name__ == '__main__':

    # Setting up an empty dataset to load the data into
    [dataset,features,labels] = loadData(dataset_path)
    nn_model = keras.models.load_model(str(model_path))

    predictions = nn_model.predict(features)


    nn_model.evaluate(features,labels)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)



    plt.figure(1)
    plt.plot(spi.cumtrapz(predictions[:,0],dx=0.0333333,axis=0))
    plt.plot(labels[:,0])
    # plt.plot(features[:,0])
    plt.title("P")
    plt.legend(["Predict","Ground Truth"])
    plt.grid()


    plt.figure(2)
    plt.plot(spi.cumtrapz(predictions[:,1],dx=0.0333333,axis=0))
    plt.plot(labels[:,1])
    # plt.plot(features[:,1])
    plt.legend(["Predict","Ground Truth"])
    plt.grid()
    plt.title("Q")

    plt.figure(3)
    plt.xlabel("Time -[$\mu$s]")
    plt.plot(spi.cumtrapz(predictions[:,2],dx=0.03333333,axis=0))
    plt.plot(labels[:,2])
    plt.title("R")
    # plt.plot(features[:,2])
    plt.grid()
    plt.legend(["Predict","Ground Truth"])


    plt.figure(4)
    plt.plot(predictions[:,3])
    plt.plot(labels[:,3])
    plt.grid()
    plt.title("Udot")
    plt.legend(["Predict","Ground Truth"])


    plt.figure(5)
    plt.plot(predictions[:,4])
    plt.plot(labels[:,4])
    plt.title("Vdot")
    plt.grid()
    plt.legend(["Predict","Ground Truth"])


    plt.figure(6)
    plt.plot(predictions[:,5])
    plt.plot(labels[:,5])
    plt.title("Wdot")
    plt.legend(["Predict","Ground Truth"])
    plt.grid()


    plt.show()
