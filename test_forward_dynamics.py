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
    features = np.zeros( (div_timeSteps,10*Nt) )   # +1 is for the input
    labels = np.zeros( (div_timeSteps,6) )
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

        data.close()

        total_steps = trange(Nt, timeSteps - Nt,div, desc='Loading time step: ', leave=True)
        i = 0
        # for t_steps in range( Nt, timesteps - Nt ):
        for step in total_steps:

            labels[i,0] = Pdot[step+div]
            labels[i,1] = Qdot[step+div]
            labels[i,2] = Rdot[step+div]
            labels[i,3] = Udot[step+div]
            labels[i,4] = Vdot[step+div]
            labels[i,5] = Wdot[step+div]

            total_steps.set_description("Loading time step (%s)" %step)
            total_steps.refresh() # to show immediately the update

            for n in range(0,Nt,div):
                features[i,n+0*Nt] = input_1[step-n*div]
                features[i,n+1*Nt] = input_2[step-n*div]
                features[i,n+2*Nt] = input_3[step-n*div]
                features[i,n+3*Nt] = input_4[step-n*div]

                features[i,n+4*Nt] = P[step-n*div]
                features[i,n+5*Nt] = Q[step-n*div]
                features[i,n+6*Nt] = R[step-n*div]

                features[i,n+7*Nt] = U[step-n*div]
                features[i,n+8*Nt] = V[step-n*div]
                features[i,n+9*Nt] = W[step-n*div]

            i += 1

    labels[:,0] = labels[:,0]/maxPdot
    labels[:,1] = labels[:,1]/maxQdot
    labels[:,2] = labels[:,2]/maxRdot

    labels[:,3] = labels[:,3]/maxUdot
    labels[:,4] = labels[:,4]/maxVdot
    labels[:,5] = labels[:,5]/maxWdot

    features[:,0*Nt:0*Nt+Nt] = features[:,0*Nt:0*Nt+Nt]/maxInput
    features[:,1*Nt:1*Nt+Nt] = features[:,1*Nt:1*Nt+Nt]/maxInput
    features[:,2*Nt:2*Nt+Nt] = features[:,2*Nt:2*Nt+Nt]/maxInput
    features[:,3*Nt:3*Nt+Nt] = features[:,3*Nt:3*Nt+Nt]/maxInput

    features[:,4*Nt:4*Nt+Nt] = features[:,4*Nt:4*Nt+Nt]/maxP
    features[:,5*Nt:5*Nt+Nt] = features[:,5*Nt:5*Nt+Nt]/maxQ
    features[:,6*Nt:6*Nt+Nt] = features[:,6*Nt:6*Nt+Nt]/maxR

    features[:,7*Nt:7*Nt+Nt] = features[:,7*Nt:7*Nt+Nt]/maxU
    features[:,8*Nt:8*Nt+Nt] = features[:,8*Nt:8*Nt+Nt]/maxV
    features[:,9*Nt:9*Nt+Nt] = features[:,9*Nt:9*Nt+Nt]/maxW

    return [labels,features,input_1,input_2,input_3,input_4,Udot,Vdot,Wdot,Pdot,Qdot,Rdot]



def getResponseFromNN(model,features,timesteps,Nt,div):
    div_timeSteps = int(timesteps/div)
    predictions = np.zeros( (div_timeSteps,6) )

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
plt.rc('font', size=12)

plt.figure(1)
plt.plot(predictions[:,0]*maxPdot,'-', mew=1, ms=8,mec='w')
plt.plot(Pdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.legend(['$\hat{\dot{P}}$', '$\dot{P}$'])
plt.title('Angular Acceleration, $\dot{P}$ of Drone ')
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('Time - [s]')
plt.ylabel('Angular Acceleration - [rad/s$^{2}$]')

plt.figure(2)
plt.plot(predictions[:,1]*maxQdot,'-', mew=1, ms=8,mec='w')
plt.plot(Qdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')
plt.legend(['$\hat{\dot{Q}}$', '$\dot{Q}$'])


plt.figure(3)
plt.plot(predictions[:,2]*maxRdot,'-', mew=1, ms=8,mec='w')
plt.plot(Rdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')
plt.legend(['$\hat{\dot{R}}$', '$\dot{R}$'])


plt.figure(4)
plt.plot(predictions[:,3]*maxUdot,'-', mew=1, ms=8,mec='w')
plt.plot(Udot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.title('Acceleration In X Directions')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('Time - [s]')
plt.legend(['$\hat{\dot{U}}$', '$\dot{U}$'])
plt.ylabel('Acceleration - [m/s$^{2}$]')

plt.figure(5)
plt.plot(predictions[:,4]*maxVdot,'-', mew=1, ms=8,mec='w')
plt.plot(Vdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')
plt.legend(['$\hat{\dot{V}}$', '$\dot{V}$'])

plt.figure(6)
plt.plot(predictions[:,5]*maxWdot,'-', mew=1, ms=8,mec='w')
plt.plot(Wdot[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')
plt.legend(['$\hat{\dot{W}}$', '$\dot{W}$'])

plt.figure(7)
plt.plot(input_1[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.plot(input_2[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.plot(input_3[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.plot(input_4[Nt:-1:div],'-', mew=1, ms=8,mec='w')
plt.grid()
plt.xticks(graph_ticks_spacing, graph_ticks_words )
plt.xlabel('time - [s]')
plt.legend(['$T_{1}$', '$T_{2}$','$T_{3}$','$T_{4}$'])

plt.figure(8)
plt.plot(predictions)
plt.plot(labels)

plt.grid()

plt.show()
