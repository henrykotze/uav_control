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
import matplotlib.pyplot as plt



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
print('----------------------------------------------------------------')
print('Fetching model info from: {}'.format(model_readme))
print('----------------------------------------------------------------')

with shelve.open(model_readme) as db:
    window_size = db['window_size']
    num_samples = db['dataset_num_entries']
    maxUdot = db['maxUdot']
    maxVdot = db['maxVdot']
    maxWdot = db['maxWdot']
    maxP = db['maxP']
    maxQ = db['maxQ']
    maxR = db['maxR']

    for key,value in db.items():
        data.append([str(key),str(value)])

db.close()

table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)

# Importing saved model
model = tf.keras.models.load_model(path_to_model)


input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [11, 12, 13, 14, 15, 16]
step = 1
test_dataset = esl_timeseries_dataset(dataset,window_size,1,1,input_indices,
                output_indices,shuffle=False,percent=0.1)


predictions = []

for (x_test, y_test) in test_dataset:
    predict = model.predict_on_batch(x_test)
    predictions.append(predict)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.plot(predictions[0,:]*maxPdot,'-', mew=1, ms=8,mec='w')
plt.plot(Pdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.legend(['$\hat{\dot{P}}$', '$\dot{P}$'])
plt.title('Angular Acceleration, $\dot{P}$ of Drone ')
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('Time - [s]')
plt.ylabel('Angular Acceleration - [rad/s$^{2}$]')

# plt.figure(2)
# plt.plot(predictions[:,1]*maxQdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Qdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$\hat{\dot{Q}}$', '$\dot{Q}$'])
#
#
# plt.figure(3)
# plt.plot(predictions[:,2]*maxRdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Rdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$\hat{\dot{R}}$', '$\dot{R}$'])
#
#
# plt.figure(4)
# plt.plot(predictions[:,3]*maxUdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Udot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.title('Acceleration In X Directions')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('Time - [s]')
# plt.legend(['$\hat{\dot{U}}$', '$\dot{U}$'])
# plt.ylabel('Acceleration - [m/s$^{2}$]')
#
# plt.figure(5)
# plt.plot(predictions[:,4]*maxVdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Vdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$\hat{\dot{V}}$', '$\dot{V}$'])
#
# plt.figure(6)
# plt.plot(predictions[:,5]*maxWdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Wdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$\hat{\dot{W}}$', '$\dot{W}$'])

plt.show()
