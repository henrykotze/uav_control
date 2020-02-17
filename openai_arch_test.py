#!/usr/bin/env python3


import tensorflow as tf
from tensorflow import keras

import numpy as np

import argparse
import sys
import shelve
import h5py
from esl_timeseries_dataset import esl_timeseries_dataset
from terminaltables import AsciiTable
from tqdm import tnrange, trange
import datetime
import os



print('----------------------------------------------------------------')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print('----------------------------------------------------------------')






def create_lstm_model(input_dims=10,batchsize=10):

    model = keras.Sequential([
    keras.layers.Dense(1024,input_shape=(batchsize,input_dims)),
    keras.layers.ReLU(),
    keras.layers.LSTM(512,input_shape=(batchsize,)),
    keras.layers.Dense(6)
    ])

    model.summary()
    return model




meep = create_lstm_model(10,10)
