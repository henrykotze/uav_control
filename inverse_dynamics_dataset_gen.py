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

parser = argparse.ArgumentParser(\
        prog='Generates a dataset from UAV responses',\
        description='Generates a H5F dataset based on .npz files containting\
         the UAV responses'
        )


parser.add_argument('-loc', default='./train_data/', help='location to stored responses')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used')
parser.add_argument('-dataset_name', default='dataset0', help='name of your dataset')
parser.add_argument('-dataset_loc', default='./datasets/', help='location to store dataset')
parser.add_argument('-div', default=1, help='')


args = parser.parse_args()

dir = vars(args)['loc']

# Getting information from readme file of training data

print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/readme'))
print('----------------------------------------------------------------')
with shelve.open( str(dir+'/readme')) as db:
    t = int(db['t'])
    dt = float(db['dt'])
    numberSims = int(db['numSim'])
    filename = db['filename']
    max_input = float(db['maxInput'])

    print("{:<15} {:<10}".format('Label','Value'))
    for key,value in db.items():
        print("{:<15} {:<10}".format(key, value))
db.close()




Nt = int(vars(args)['Nt'])
div = int(vars(args)['div'])
timeSteps = int(t/dt)
nameOfDataset = str(vars(args)['dataset_name'])
dataset_loc = str(vars(args)['dataset_loc'])
div_timeSteps = int(timeSteps/div)



with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)

    with shelve.open(str(dir+'/readme')) as data_readme:
        for key in data_readme:
            db[key] = data_readme[key]

    data_readme.close()
db.close()

if __name__ == '__main__':

    # Pre-creating correct sizes of arrays
    features = np.zeros( (div_timeSteps*numberSims,18*Nt))    # +1 is for the inpu
    labels = np.zeros((div_timeSteps*numberSims,4))

    maxP = 0
    maxQ = 0
    maxR = 0

    maxPdot = 0
    maxQdot = 0
    maxRdot = 0

    maxU = 0
    maxV = 0
    maxW = 0

    maxUdot = 0
    maxVdot = 0
    maxWdot = 0

    maxL = 0
    maxM = 0
    maxN = 0

    maxX = 0
    maxY = 0
    maxZ = 0

    maxTorque = 0


    numSim = trange(numberSims, desc='Loading from: ', leave=True)
    for numFile in numSim:

        numSim.set_description("Loading From (%s)" %filename)
        numSim.refresh()

        with np.load(str(dir+'/'+filename)) as data:

            input1 = data['input_1']
            input2 = data['input_2']
            input3 = data['input_3']
            input4 = data['input_4']

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

            if(np.amax(U) > maxU):
                 maxU = np.amax(U)

            if(np.amax(V) > maxV):
                 maxV = np.amax(V)

            if(np.amax(W) > maxW):
                 maxW = np.amax(W)

            if(np.amax(Udot) > maxUdot):
                 maxUdot = np.amax(Udot)

            if(np.amax(Vdot) > maxVdot):
                 maxVdot = np.amax(Vdot)

            if(np.amax(Wdot) > maxWdot):
                 maxWdot = np.amax(Wdot)


            if(np.amax(P) > maxP):
                 maxP = np.amax(P)

            if(np.amax(Q) > maxQ):
                 maxQ = np.amax(Q)

            if(np.amax(R) > maxR):
                 maxR = np.amax(R)

            if(np.amax(Pdot) > maxPdot):
                 maxPdot = np.amax(Pdot)

            if(np.amax(Qdot) > maxQdot):
                 maxQdot = np.amax(Qdot)

            if(np.amax(Rdot) > maxRdot):
                 maxRdot = np.amax(Rdot)

            if(np.amax(L) > maxL):
                 maxL = np.amax(L)

            if(np.amax(M) > maxM):
                 maxM = np.amax(M)

            if(np.amax(N) > maxN):
                 maxN = np.amax(N)

            if(np.amax(X) > maxX):
                 maxX = np.amax(X)

            if(np.amax(Y) > maxY):
                 maxY = np.amax(Y)

            if(np.amax(Z) > maxZ):
                 maxZ = np.amax(Z)


            i = 0
            for step in range( Nt, timeSteps - Nt,div):

                labels[i+div_timeSteps*numFile,0] = input1[step]
                labels[i+div_timeSteps*numFile,1] = input2[step]
                labels[i+div_timeSteps*numFile,2] = input3[step]
                labels[i+div_timeSteps*numFile,3] = input4[step]

                for n in range(0,Nt,div):

                    features[i+div_timeSteps*numFile,n+0*Nt] = Pdot[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+1*Nt] = Qdot[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+2*Nt] = Rdot[step-n*div+div]

                    features[i+div_timeSteps*numFile,n+3*Nt] = Udot[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+4*Nt] = Vdot[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+5*Nt] = Wdot[step-n*div+div]

                    features[i+div_timeSteps*numFile,n+6*Nt] = P[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+7*Nt] = Q[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+8*Nt] = R[step-n*div+div]

                    features[i+div_timeSteps*numFile,n+9*Nt] =  U[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+10*Nt] = V[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+11*Nt] = W[step-n*div+div]

                    features[i+div_timeSteps*numFile,n+12*Nt] = X[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+13*Nt] = Y[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+14*Nt] = Z[step-n*div+div]

                    features[i+div_timeSteps*numFile,n+15*Nt] = L[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+16*Nt] = M[step-n*div+div]
                    features[i+div_timeSteps*numFile,n+17*Nt] = N[step-n*div+div]

                i = i + 1

            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))


    labels[:,0] = labels[:,0]/max_input
    labels[:,1] = labels[:,1]/max_input
    labels[:,2] = labels[:,2]/max_input
    labels[:,3] = labels[:,3]/max_input

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


    with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme')) as db:
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

        db['maxL'] = maxL
        db['maxM'] = maxM
        db['maxN'] = maxN

        db['maxX'] = maxX
        db['maxY'] = maxY
        db['maxZ'] = maxZ

        db['maxTorque'] = maxTorque

    db.close()


    print('\n--------------------------------------------------------------')
    print('Saving features and labels to:', nameOfDataset)
    print('--------------------------------------------------------------')

    h5f = h5py.File(str(dataset_loc + '/'+nameOfDataset),'w')
    h5f.create_dataset('features', data=features)
    h5f.create_dataset('labels', data=labels)
    h5f.close()
