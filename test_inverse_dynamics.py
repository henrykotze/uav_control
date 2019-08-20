#!/usr/bin/env python3
import numpy as np
import sys
import argparse
import tensorflow as tf
import os
import pickle
import shelve
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uav_model import drone
from tqdm import trange
np.set_printoptions(threshold=sys.maxsize)
parser = argparse.ArgumentParser(\
        prog='Test the performance of the forward dynamics',\
        description='Environment where the trained neural network is tested'
        )


parser.add_argument('-model_path', default='./nn_mdl', help='path to neural network model')
parser.add_argument('-n', default=0, help='Trained response to test')
parser.add_argument('-loc', default=0, help='Trained response to test')


args = parser.parse_args()


dir = str(vars(args)['loc'])
filename = str(vars(args)['loc'])+'/'+'respone-0.npz'
model_path = vars(args)['model_path']
number = vars(args)['n']



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
print('Fetching training info from: ', model_readme)
print('----------------------------------------------------------------')
with shelve.open(model_readme) as db:
    t = int((db)['t'])
    dt = float((db)['dt'])
    Nt = int((db)['Nt'])
    div = int((db)['div'])
    maxQdot = float((db)['maxQdot'])
    maxPdot = float((db)['maxPdot'])
    maxRdot = float((db)['maxRdot'])
    maxVdot = float((db)['maxVdot'])
    maxUdot = float((db)['maxUdot'])
    maxWdot = float((db)['maxWdot'])
    maxQ = float((db)['maxQ'])
    maxP = float((db)['maxP'])
    maxR = float((db)['maxR'])
    maxV = float((db)['maxV'])
    maxU = float((db)['maxU'])
    maxW = float((db)['maxW'])


    maxX = float((db)['maxX'])
    maxY = float((db)['maxY'])
    maxZ = float((db)['maxZ'])
    maxL = float((db)['maxL'])
    maxM = float((db)['maxM'])
    maxN = float((db)['maxN'])

    train_dir = str((db)['loc'])
    filename = train_dir + '/' + str((db)['filename'])
    maxInput = float((db)['maxInput'])
db.close()


print('----------------------------------------------------------------')
print('Training Information: ')
print('----------------------------------------------------------------')
with shelve.open(model_readme) as db:
    # for key,value in db.items():
    print("{:<30} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<30} {:<10}".format(key, value))
db.close()
print('----------------------------------------------------------------')



def getFeaturesAndResponse(filename,Nt,timeSteps,div):
    div_timeSteps = int(timeSteps/div)
    features = np.zeros( (div_timeSteps,18*Nt) )   # +1 is for the input
    labels = np.zeros( (div_timeSteps,4) )
    with np.load(filename) as data:

        print('----------------------------------------------------------------')
        print('Loading Data from: ', filename)
        print('----------------------------------------------------------------')

        # data = np.load(filename)

        input_1 = data['input_1']
        input_2 = data['input_2']
        input_3 = data['input_3']
        input_4 = data['input_4']

        U = data['U']
        V = data['V']
        W = data['W']

        Udot = data['Udot']
        Vdot = data['Vdot']
        Wdot = data['Wdot']

        P = data['P']
        Q = data['Q']
        R = data['R']

        Pdot = data['Pdot']
        Qdot = data['Qdot']
        Rdot = data['Rdot']

        Pdot = data['Pdot']
        Qdot = data['Qdot']
        Rdot = data['Rdot']

        X = data['X']
        Y = data['Y']
        Z = data['Z']

        L = data['L']
        M = data['M']
        N = data['N']

        data.close()

        total_steps = trange(Nt, timeSteps - Nt,div, desc='Loading time step: ', leave=True)
        i = 0
        # for t_steps in range( Nt, timesteps - Nt ):
        for step in total_steps:

            labels[i,0] = input_1[step]
            labels[i,1] = input_2[step]
            labels[i,2] = input_3[step]
            labels[i,3] = input_4[step]

            total_steps.set_description("Loading time step (%s)" %step)
            total_steps.refresh() # to show immediately the update

            for n in range(0,Nt,div):

                features[i,n+0*Nt] = Pdot[step-n*div+div]
                features[i,n+1*Nt] = Qdot[step-n*div+div]
                features[i,n+2*Nt] = Rdot[step-n*div+div]

                features[i,n+3*Nt] = Udot[step-n*div+div]
                features[i,n+4*Nt] = Vdot[step-n*div+div]
                features[i,n+5*Nt] = Wdot[step-n*div+div]

                features[i,n+6*Nt] = P[step-n*div+div]
                features[i,n+7*Nt] = Q[step-n*div+div]
                features[i,n+8*Nt] = R[step-n*div+div]

                features[i,n+9*Nt] =  U[step-n*div+div]
                features[i,n+10*Nt] = V[step-n*div+div]
                features[i,n+11*Nt] = W[step-n*div+div]

                features[i,n+12*Nt] = X[step-n*div+div]
                features[i,n+13*Nt] = Y[step-n*div+div]
                features[i,n+14*Nt] = Z[step-n*div+div]

                features[i,n+15*Nt] = L[step-n*div+div]
                features[i,n+16*Nt] = M[step-n*div+div]
                features[i,n+17*Nt] = N[step-n*div+div]

            i += 1

    labels[:,0] = labels[:,0]/maxInput
    labels[:,1] = labels[:,1]/maxInput
    labels[:,2] = labels[:,2]/maxInput
    labels[:,3] = labels[:,3]/maxInput

    features[:,0*Nt:0*Nt+Nt] = features[:,0*Nt:0*Nt+Nt]/maxPdot
    features[:,1*Nt:1*Nt+Nt] = features[:,1*Nt:1*Nt+Nt]/maxQdot
    features[:,2*Nt:2*Nt+Nt] = features[:,2*Nt:2*Nt+Nt]/maxRdot

    features[:,3*Nt:3*Nt+Nt] = features[:,3*Nt:3*Nt+Nt]/maxUdot
    features[:,4*Nt:4*Nt+Nt] = features[:,4*Nt:4*Nt+Nt]/maxVdot
    features[:,5*Nt:5*Nt+Nt] = features[:,5*Nt:5*Nt+Nt]/maxWdot

    features[:,6*Nt:6*Nt+Nt] = features[:,6*Nt:6*Nt+Nt]/maxP
    features[:,7*Nt:7*Nt+Nt] = features[:,7*Nt:7*Nt+Nt]/maxQ
    features[:,8*Nt:8*Nt+Nt] = features[:,8*Nt:8*Nt+Nt]/maxR

    features[:,9*Nt:9*Nt+Nt] = features[:,9*Nt:9*Nt+Nt]/maxU
    features[:,10*Nt:10*Nt+Nt] = features[:,10*Nt:10*Nt+Nt]/maxV
    features[:,11*Nt:11*Nt+Nt] = features[:,11*Nt:11*Nt+Nt]/maxR

    features[:,12*Nt:12*Nt+Nt] = features[:,12*Nt:12*Nt+Nt]/maxX
    features[:,13*Nt:13*Nt+Nt] = features[:,13*Nt:13*Nt+Nt]/maxY
    features[:,14*Nt:14*Nt+Nt] = features[:,14*Nt:14*Nt+Nt]/maxZ

    features[:,15*Nt:15*Nt+Nt] = features[:,15*Nt:15*Nt+Nt]/maxL
    features[:,16*Nt:16*Nt+Nt] = features[:,16*Nt:16*Nt+Nt]/maxM
    features[:,17*Nt:17*Nt+Nt] = features[:,17*Nt:17*Nt+Nt]/maxN

    return [labels,features,input_1,input_2,input_3,input_4,Udot,Vdot,Wdot,Pdot,Qdot,Rdot]



def getResponseFromNN(model,features,timesteps,Nt,div):
    div_timeSteps = int(timesteps/div)
    predictions = np.zeros( (div_timeSteps,4) )

    print('------------------------------------------------------------------------------------------------------')
    print('Fetching Response From Neural Network: ')
    print('------------------------------------------------------------------------------------------------------')

    total_steps = trange(Nt, div_timeSteps-Nt, desc='time step: ', leave=True)

    for t_steps in total_steps:
        total_steps.set_description("time step (%s)" %t_steps)
        total_steps.refresh() # to show immediately the update
        predictions[t_steps] = model.predict( np.array( [features[t_steps,:] ]) )
        # print(features[t_steps,:])
    return predictions

print('------------------------------------------------------------------------------------------------------')
print('Fetching neural network model from: ', str(model_path ))
print('------------------------------------------------------------------------------------------------------')

nn_model = keras.models.load_model(str(model_path))
inputsize = nn_model.get_input_shape_at(0)[1]
filename = filename.replace(str(0),str(number))
input_matrix = np.zeros((1,inputsize))
timeSteps = int(t/dt)



[labels,features,input_1,input_2,input_3,input_4,Udot,Vdot,Wdot,Pdot,Qdot,Rdot] = \
getFeaturesAndResponse(filename,Nt,timeSteps,div)


predictions = getResponseFromNN(nn_model,features,timeSteps,Nt,div)

graph_ticks_spacing = np.arange(0,int(timeSteps/div)+int(1/(div*dt)),int(1/(div*dt)))
graph_ticks_words = np.arange(0,t+1,1).tolist()


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14)

plt.figure(1)
plt.plot(predictions[:,0],'-', mew=1, ms=8,mec='w')
plt.plot(labels[:,0],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.legend(['$\hat{\dot{P}}$', '$\dot{P}$'])
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')

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
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$\hat{\dot{U}}$', '$\dot{U}$'])
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
#
# plt.figure(7)
# plt.plot(input_1[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.plot(input_2[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.plot(input_3[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.plot(input_4[Nt:-1:div],'-', mew=1, ms=8,mec='w')
# plt.grid()
# plt.xticks(graph_ticks_spacing, graph_ticks_words )
# plt.xlabel('time - [s]')
# plt.legend(['$T_{1}$', '$T_{2}$','$T_{3}$','$T_{4}$'])
#
# plt.figure(8)
# plt.plot(predictions)
# plt.plot(labels)


plt.show()
