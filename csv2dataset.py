#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import numpy as np
import csv
import sys
import argcomplete, argparse
import tensorflow as tf
import math
import os
import pickle
import shelve
import pandas as pd
import h5py
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
parser.add_argument('-dataset_name', default=0, help='Test on measured data')
parser.add_argument('-loc1', default=0, help='csv 1')
parser.add_argument('-loc2', default=0, help='csv 2')
parser.add_argument('-loc3', default=0, help='csv 2')


# args = parser.parse_args()
argcomplete.autocomplete(parser)
args = parser.parse_args()


dataset_name = str(vars(args)['dataset_name'])
model_path = vars(args)['model_path']
loc1 = str(vars(args)['loc1'])
loc2 = str(vars(args)['loc2'])
loc3 = str(vars(args)['loc3'])




def convertInertia2Body(q,inertia):

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]


    r11 = q0**2 + q1**2 + q2**2 +q3**2
    r12 = 2*(q1*q2 +q0*q3)
    r13 = 2*(q1*q3 - q0*q2)

    r21 = 2*(q1*q2-q0*q3)
    r22 = q0**2 - q1**2 + q2**2 - q3**2
    r23 = 2*(q2*q3 + q0*q1)

    r31 = 2*(q1*q3 + q0*q2)
    r32 = 2*(q2*q3 - q0*q1)
    r33 = q0**2 - q1**2 - q2**2 + q3**2

    # print(inertia[0].size)

    U = r11*inertia[0] + r12*inertia[1] + r13*inertia[2]
    V = r21*inertia[0] + r22*inertia[1] + r23*inertia[2]
    W = r31*inertia[0] + r32*inertia[1] + r33*inertia[2]

    return [U,V,W]



def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        # return array[idx-1]
        return idx-1
    else:
        # return array[idx]
        return idx


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
    dt = float((db)['dt'])
    Nt = int((db)['Nt'])
    div = int((db)['div'])
    maxQdot = float((db)['maxQdot'])
    maxInput = float((db)['maxInput'])
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
    dataset_loc = str((db)['dataset'])
    # filename = train_dir + '/' + str((db)['filename'])
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



# vechile_attitude
df = pd.read_csv(loc1)

timestamp_attitude = np.array(df['timestamp'].tolist())
P = np.array(df['rollspeed'].tolist()).astype(float)
Q = np.array(df['pitchspeed'].tolist()).astype(float)
R = np.array(df['yawspeed'].tolist()).astype(float)
q1 = np.array(df['q[0]'].tolist()).astype(float)
q2 = np.array(df['q[1]'].tolist()).astype(float)
q3 = np.array(df['q[2]'].tolist()).astype(float)
q4 = np.array(df['q[3]'].tolist()).astype(float)
q = [q1,q2,q3,q4]
length_attitude = timestamp_attitude.size


# vechile_local_position
df = pd.read_csv(loc2)
timestamp_local_position = np.array(df['timestamp'].tolist())
inertia_U = np.array(df['vx'].tolist()).astype(float)
inertia_V = np.array(df['vy'].tolist()).astype(float)
inertia_W = np.array(df['vz'].tolist()).astype(float)
inertia_Udot = np.array(df['ax'].tolist()).astype(float)
inertia_Vdot = np.array(df['ay'].tolist()).astype(float)
inertia_Wdot = np.array(df['az'].tolist()).astype(float)


# vechile_actuator_outputs
df = pd.read_csv(loc3)
timestamp_actuator_outputs = np.array(df['timestamp'].tolist())
timestamp_actuator_outputs_cor = np.zeros(length_attitude)
PWM_0 = np.array(df['output[0]'].tolist()).astype(float)
PWM_1 = np.array(df['output[1]'].tolist()).astype(float)
PWM_2 = np.array(df['output[2]'].tolist()).astype(float)
PWM_3 = np.array(df['output[3]'].tolist()).astype(float)



P_ = np.zeros(len(timestamp_local_position))
Q_ = np.zeros(len(timestamp_local_position))
R_ = np.zeros(len(timestamp_local_position))
q1_ = np.zeros(len(timestamp_local_position))
q2_ = np.zeros(len(timestamp_local_position))
q3_ = np.zeros(len(timestamp_local_position))
q4_ = np.zeros(len(timestamp_local_position))



time_keeper = 0

for value in timestamp_local_position:
    loc = find_nearest(timestamp_attitude,value)
    P_[time_keeper] = P[loc]
    Q_[time_keeper] = Q[loc]
    R_[time_keeper] = R[loc]
    q1_[time_keeper] = q1[loc]
    q2_[time_keeper] = q2[loc]
    q3_[time_keeper] = q3[loc]
    q4_[time_keeper] = q4[loc]

    time_keeper += 1

q_ = [q1_,q2_,q3_,q4_]


inertia_Vel = [inertia_U,inertia_V,inertia_W]
inertia_Acc = [inertia_Udot,inertia_Vdot,inertia_Wdot]

[U_,V_,W_] = convertInertia2Body(q_,inertia_Vel)
[Udot,Vdot,Wdot] = convertInertia2Body(q_,inertia_Acc)


T1 = (PWM_0/1000 - 1)*3.2*9.81
T2 = (PWM_1/1000 - 1)*3.2*9.81
T3 = (PWM_2/1000 - 1)*3.2*9.81
T4 = (PWM_3/1000 - 1)*3.2*9.81

timeSteps = len(timestamp_local_position)




features = np.zeros((timeSteps,10*Nt))
labels = np.zeros((timeSteps,6))

div = 1
i = 0
for step in range( Nt, timeSteps - Nt):

    labels[i,0] = P_[step+div]
    labels[i,1] = Q_[step+div]
    labels[i,2] = R_[step+div]

    labels[i,3] = Udot[step]
    labels[i,4] = Vdot[step]
    labels[i,5] = Wdot[step]

    for n in range(0,Nt,div):

        features[i,n+0*Nt] = T1[step]
        features[i,n+1*Nt] = T2[step]
        features[i,n+2*Nt] = T3[step]
        features[i,n+3*Nt] = T4[step]
        features[i,n+4*Nt] = P_[step]
        features[i,n+5*Nt] = Q_[step]
        features[i,n+6*Nt] = R_[step]
        features[i,n+7*Nt] = U_[step]
        features[i,n+8*Nt] = V_[step]
        features[i,n+9*Nt] = W_[step]
    i = i + 1

# labels[:,0] = labels[:,0]/maxPdot
# labels[:,1] = labels[:,1]/maxQdot
# labels[:,2] = labels[:,2]/maxRdot

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




with shelve.open( str(dataset_loc + '/'+dataset_name+'_readme')) as db:
    db['maxPdot'] = maxPdot
    db['maxQdot'] = maxQdot
    db['maxRdot'] = maxRdot

    db['maxUdot'] = maxUdot
    db['maxVdot'] = maxVdot
    db['maxWdot'] = maxWdot

    db['maxP'] = maxP
    db['maxQ'] = maxQ
    db['maxR'] = maxR

    db['maxU'] = maxU
    db['maxV'] = maxV
    db['maxW'] = maxW
    db['timeSteps'] = timeSteps
    db['Nt'] = Nt

    print("{:<15} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<15} {:<10}".format(key, value))
db.close()



print('\n--------------------------------------------------------------')
print('Saving features and labels to:', dataset_name)
print('--------------------------------------------------------------')

h5f = h5py.File(str(dataset_loc + '/'+dataset_name),'w')
h5f.create_dataset('features', data=features)
h5f.create_dataset('labels', data=labels)
h5f.close()





plt.figure(1)
plt.plot(P_,'.-')
plt.title('P')
plt.grid()


plt.figure(2)
plt.plot(Q_,'.-')
plt.title('Q')
plt.grid()


plt.figure(3)
plt.plot(R_,'.-')
plt.title("R")
plt.grid()






plt.figure(4)
# plt.plot(timestamp_local_position_cor,'.-')
# plt.plot(timestamp_actuator_outputs_cor,'.-')
plt.plot(Udot,'.-')
plt.title("Udot")
plt.grid()


plt.figure(5)
# plt.plot(timestamp_local_position_cor,'.-')
# plt.plot(timestamp_actuator_outputs_cor,'.-')
plt.plot(Vdot,'.-')
plt.title("Vdot")
plt.grid()


plt.figure(6)
# plt.plot(timestamp_local_position_cor,'.-')
# plt.plot(timestamp_actuator_outputs_cor,'.-')
plt.plot(Wdot,'.-')
plt.title("Wdot")
plt.grid()


plt.figure(7)
plt.plot(T1)
plt.plot(T2)
plt.plot(T3)
plt.plot(T4)
plt.grid()
plt.title("Torque")



plt.show()

#
