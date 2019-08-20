#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random as rand
import argparse
import os
import pickle
import shelve
import h5py
from tqdm import trange
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(\
        prog='Generates a dataset of UAV responses',\
        description='Generates a H5F dataset based on .npz files containting\
         the UAV responses'
        )


parser.add_argument('-dataset_name', default='dataset0', help='name of your dataset')
parser.add_argument('-dataset_loc', default='./datasets/', help='location to store dataset')
parser.add_argument('-n', default=1, help='')

args = parser.parse_args()

dir = vars(args)['dataset_loc']
dataset = vars(args)['dataset_name']
dataset_path = dir + '/' + dataset
n = int(vars(args)['n'])


with shelve.open( str(dataset_path+'_readme') ) as db:
    # system = db['system']
    t = int(db['t'])
    numberSims = int(db['numSim'])
    filename = db['filename']
    div  = int(db['div'])
    Nt  = int(db['Nt'])
    dt  = float(db['dt'])

    print("{:<15} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<15} {:<10}".format(key, value))
db.close()


timeSteps = int(t/dt)
div_timesteps = int(timeSteps/div)


def loadData(dir,n,timesteps):
    # in the directory, dir, determine how many *.npz files it contains
    with h5py.File(str(dir),'r') as h5f:
        print('==============================================================')
        print('Loading dataset from: ' ,str(dir))
        print('==============================================================\n')
        features = h5f['features'][:]
        labels = h5f['labels'][:]
        h5f.close()

    return labels[timesteps*n:timesteps*n+timesteps,0]



feature = loadData(dataset_path,n,div_timesteps)




plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)


# plt.figure(10)
# plt.title('Torque by motors')
# plt.plot(features,'.-', mew=1, ms=8,mec='w')
# plt.plot(Pdot,'-', mew=1, ms=8,mec='w')

plt.figure(1)
plt.plot(feature,'-', mew=1, ms=8,mec='w')
plt.show()
