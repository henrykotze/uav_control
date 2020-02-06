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
        prog='Test the feedforward neural network for the forward dynamics of a drone',\
        description=''
        )


parser.add_argument('-model', default='', help='path to feedforward neural network')
parser.add_argument('-percent', default=0.1, help='percentage of dataset to test and display plots')
parser.add_argument('-dataset', default=0.1, help='path to dataset to test')


args = parser.parse_args()

path_to_model = str(vars(args)['model'])
percent = float(vars(args)['percent'])
dataset = str(vars(args)['dataset'])

dataset_readme = dataset + '_readme'
model_readme  = path_to_model + '_readme'

print('----------------------------------------------------------------')
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print('----------------------------------------------------------------')

data=[]
print('----------------------------------------------------------------')
print('Fetching dataset info from: {}'.format(dataset_readme))
print('----------------------------------------------------------------')
with shelve.open(dataset_readme) as db:
    for key,value in db.items():
        data.append([str(key),str(value)])
db.close()
table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)

data =[]
print('----------------------------------------------------------------')
print('Fetching model info from: {}'.format(model_readme))
print('----------------------------------------------------------------')
with shelve.open(model_readme) as db:
    window_size = db['window_size']
    num_samples = db['dataset_num_entries']

    for key,value in db.items():
        data.append([str(key),str(value)])
db.close()
table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)

# Importing saved model



input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [11, 12, 13, 14, 15, 16]
step = 1
test_dataset = esl_timeseries_dataset(dataset,window_size,1,1,input_indices,
                output_indices,shuffle=False,percent=0.1)
#
#
#
#
for (x_test, y_test) in test_dataset:
    # predictions = model.predict_on_batch(x_test)
    print(x_test)
