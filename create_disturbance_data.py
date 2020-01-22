#!/usr/bin/env python3


import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse
from uav_model import drone
import os
import pickle
import shelve
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
from tqdm import trange

parser = argparse.ArgumentParser(\
        prog='Generates responses of a rotary wing UAV',\
        description='Creates .npz files containing the responses of a rotary wing UAV'
        )


parser.add_argument('-loc', default='./train_data/', help='location to store responses')
parser.add_argument('-filename', default="response-0.npz", help='filename')
parser.add_argument('-t', default=10, help='time lenght of responses')
parser.add_argument('-numSim', default=2, help='number of responses to generate')
parser.add_argument('-inputTime', default=50, help='time at which inputs starts')
parser.add_argument('-dt', default=0.01, help='timestep increments of response')
parser.add_argument('-maxInput', default=0.5, help='maximum input given to system')
parser.add_argument('-minInput', default=0.0, help='minimum input given to system')
parser.add_argument('-drone', default='./drone_info.txt', help='path to file containing drone info')
parser.add_argument('-max_freq', default=30, help='max disturbance freq')
parser.add_argument('-min_freq', default=15, help='min disturbance freq')



args = parser.parse_args()

t=int(vars(args)['t'])
numberSims = int(vars(args)['numSim'])
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
inputTime = int(vars(args)['inputTime'])
dt = float(vars(args)['dt'])
maxInput = float(vars(args)['maxInput'])
minInput = float(vars(args)['minInput'])
drone_file_path = str(vars(args)['drone'])
min_freq = int(vars(args)['min_freq'])
max_freq = int(vars(args)['max_freq'])

drone_file = open(drone_file_path,"r")
drone_str = drone_file.readlines()[2].split(";")[0:-1]
drone_info = [float(i) for i in drone_str]
drone_file.close()

# Add a Readme file in directory to show selected variables that describe the
# responses

# filename = filename.replace(str(0),str(0))


with shelve.open( str(dir+'/readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)
db.close()


def straightline_func(x, a, b):
    return a*x+b

def exponential_func(x, a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            y= a*np.exp(b*x)
            return y
        except:
            return 0*x

def square_func(x,a,b,c):
    return a*np.power(x,2) + b*x + c


def generateDisturbance(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    labels = np.zeros([responseDuration,max_freq-min_freq+1+1])
    # zeroInputDur = int(responseDuration/10*(np.random.random())    ) # Duration of zero input
    startInput = int(np.random.randint(0,responseDuration))
    timestep = startInput


    freq =  np.random.randint(min_freq,max_freq+1)
    inputDur = int(responseDuration-startInput)
    freq_content = np.zeros([1,max_freq-min_freq+1+1])
    no_disturb = np.zeros([1,max_freq-min_freq+1+1])
    no_disturb[0,0] = 1;
    # print(freq_content)
    # print(freq,freq-min_freq)
    freq_content[0,freq-min_freq+1] = 1

    magInput = (2)*np.random.random() # Magnitude Size of Input

    labels[0:startInput][:] = no_disturb;

    t = np.arange(timestep,timestep+inputDur)
    input[timestep:timestep+inputDur] = np.transpose(np.array([magInput*np.sin(2*np.pi*freq*t/inputDur)]))
    labels[timestep:timestep+inputDur][:] = freq_content

    return input,labels

def generateStepInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        magInput = (maxInput-minInput)*np.random.random()+minInput # Magnitude Size of Input
        inputDur = int(responseDuration/10*(np.random.random() ) ) # Duration of input
        zeroInputDur = int(responseDuration/10*(np.random.random()) ) # Duration of zero input


        input[timestep:timestep+inputDur] = magInput
        timestep += inputDur
        input[timestep:timestep+zeroInputDur] = 0
        timestep += zeroInputDur

    return input

def generateRampInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:
        magInput = (maxInput-minInput)*np.random.random()+minInput # peak point in ramp
        firstDur = int(responseDuration/10*(np.random.random()))+1 # Duration of first half ramp
        secondDur = int(responseDuration/10*(np.random.random()))+1 # Duration of second half ramp

        if(timestep + firstDur+secondDur < responseDuration):

            grad1 = magInput/firstDur   # gradient of first part
            grad2 = -magInput/secondDur  # Gradientr of second part

            firstLine = np.arange(firstDur)*grad1

            secondLine = -1*np.arange(secondDur,0,-1)*grad2
            input[timestep:timestep+firstDur] = np.transpose(np.array([firstLine]))
            timestep += firstDur
            input[timestep:timestep+secondDur] = np.transpose(np.array([secondLine]))
            timestep += secondDur
        else:
            break

    # input = addNoise(input,250)
    return input


def generateSquareInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:

        y1 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y2 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y3 = ((maxInput-minInput)*np.random.random()+minInput)/3
        y4 = ((maxInput-minInput)*np.random.random()+minInput)/3

        x1 = int(responseDuration/10*(np.random.random()))+1
        x2 = int(responseDuration/10*(np.random.random()))+1
        x3 = int(responseDuration/10*(np.random.random()))+1

        x  = np.array([timestep+1,timestep+x1,timestep+x1+x2,timestep+x1+x2+x3])
        y = np.array([y1,y1+y2,y1+y2+y3,y1+y2+y3+y4])

        if(timestep + x1 + x2 +x3 < responseDuration):

            popt, pcov = curve_fit(square_func, x, y)
            c = popt[2]
            b = popt[1]
            a = popt[0]
            curve = np.arange(timestep,timestep+x1+x2+x3)
            curve = square_func(curve, a, b, c)
            input[timestep:timestep+x1+x2+x3] = np.transpose(np.array([curve]))
            timestep = timestep + x1 + x2 + x3

        else:
            break

    # input = addNoise(input,250)
    return input

def generateExpoInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput
    error_check = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while timestep < responseDuration:


            y11 = ((maxInput-minInput)*np.random.random()+minInput)/2 #
            y12 = ((maxInput-minInput)*np.random.random()+minInput)/2 #

            x11 = int(responseDuration/10*(np.random.random())) #
            x12 = int(responseDuration/10*(np.random.random()))+20 #

            y21 = ((maxInput-minInput)*np.random.random()+minInput)/2 #

            x21 = int(responseDuration/10*(np.random.random())) #
            x22 = int(responseDuration/10*(np.random.random()))+20 #

            if(timestep + x11+ x12 + x21 + x22 + 1 < responseDuration):

                y = np.log(np.array([0.00001, y11, y11+y12]))
                x = np.array([timestep+1,timestep+x11,timestep+x11+x12])
                popt, pcov = curve_fit(straightline_func, x, y)
                b = popt[0]
                a = np.exp(popt[1])
                curve = np.arange(timestep,timestep+x11+x12)
                curve = exponential_func(curve, a, b)
                input[timestep:timestep+x11+x12] = np.transpose(np.array([curve]))

                y = np.log(np.array([y11+y12,y21, 0.0001]))
                x = np.array([timestep+x11+x12,timestep+x11+x12+x21, timestep+x11+x12+x21+x22])
                popt, pcov = curve_fit(straightline_func, x, y)
                b = popt[0]
                a = np.exp(popt[1])
                curve = np.arange(timestep+x11+x12,timestep+x11+x12+x21+x22)
                curve = exponential_func(curve, a, b)
                input[timestep+x11+x12:timestep+x11+x12+x21+x22] = np.transpose(np.array([curve]))
                timestep = timestep + x11 + x12 + x21 + x22
            else:
                break
    return input

def generateNoiseInput(responseDuration,startInput,minInput,maxInput):

    input = np.zeros( (responseDuration,1) )
    input += (maxInput-minInput)*np.random.random((np.size(input),1))+minInput
    return input

def addNoise(response,level):
    sizeOfArray = np.size(response)
    response += np.random.random((sizeOfArray,1))/level
    return response

def generateCombinationInput(responseDuration,startInput,minInput,maxInput):
    input1 = generateStepInput(responseDuration,startInput,minInput/4,maxInput/4)
    input2 = generateRampInput(responseDuration,startInput,minInput/4,maxInput/4)
    input3 = generateExpoInput(responseDuration,startInput,minInput/4,maxInput/4)
    input4 = generateSquareInput(responseDuration,startInput,minInput/4,maxInput/4)
    input = addNoise(input1+input2+input3+input4,500)
    return input


if __name__ == '__main__':
    # print('Creating the response of ', str(system_info))
    # print('Writing responses to:', filename )
    simError = 0

    timeSteps= int(t/dt) # time in number of step
    numSim = trange(numberSims, desc='# of response', leave=True)
    i = 0
    while i < numberSims:
    # for i in numSim:
        numSim.set_description("# of response (%s)" %filename)
        numSim.refresh() # to show immediately the update
        # print('Number of responses: ', numSim)
        # response = pendulum(wn,zeta,y=initial*np.pi/180,time_step=dt)
        response = drone(sys_const=drone_info,time_step=dt)
        disturbed_response = drone(sys_const=drone_info,time_step=dt)

        # Generate a random input to all 4 different motors
        input_1 = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)
        input_2 = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)
        input_3 = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)
        input_4 = generateCombinationInput(timeSteps,inputTime,minInput,maxInput)

        [disturbance,disturbance_labels] = generateDisturbance(timeSteps,inputTime,minInput,maxInput)
        # plt.plot(disturbance)
        # plt.show()

#################### UNDISTURBED ###############################################

        U = np.zeros( (timeSteps,1) )
        V = np.zeros( (timeSteps,1) )
        W = np.zeros( (timeSteps,1) )

        Udot = np.zeros( (timeSteps,1) )
        Vdot = np.zeros( (timeSteps,1) )
        Wdot = np.zeros( (timeSteps,1) )

        P = np.zeros( (timeSteps,1) )
        Q = np.zeros( (timeSteps,1) )
        R = np.zeros( (timeSteps,1) )

        Pdot = np.zeros( (timeSteps,1) )
        Qdot = np.zeros( (timeSteps,1) )
        Rdot = np.zeros( (timeSteps,1) )


        X = np.zeros( (timeSteps,1) )
        Y = np.zeros( (timeSteps,1) )
        Z = np.zeros( (timeSteps,1) )


        M = np.zeros( (timeSteps,1) )
        N = np.zeros( (timeSteps,1) )
        L = np.zeros( (timeSteps,1) )

##################### DISTURBANCE ###########################################

        U_disturb = np.zeros( (timeSteps,1) )
        V_disturb = np.zeros( (timeSteps,1) )
        W_disturb = np.zeros( (timeSteps,1) )

        Udot_disturb = np.zeros( (timeSteps,1) )
        Vdot_disturb = np.zeros( (timeSteps,1) )
        Wdot_disturb = np.zeros( (timeSteps,1) )

        P_disturb = np.zeros( (timeSteps,1) )
        Q_disturb = np.zeros( (timeSteps,1) )
        R_disturb = np.zeros( (timeSteps,1) )

        Pdot_disturb = np.zeros( (timeSteps,1) )
        Qdot_disturb = np.zeros( (timeSteps,1) )
        Rdot_disturb = np.zeros( (timeSteps,1) )

        X_disturb = np.zeros( (timeSteps,1) )
        Y_disturb = np.zeros( (timeSteps,1) )
        Z_disturb = np.zeros( (timeSteps,1) )

        M_disturb = np.zeros( (timeSteps,1) )
        N_disturb = np.zeros( (timeSteps,1) )
        L_disturb = np.zeros( (timeSteps,1) )

        t = 0

        while t < timeSteps:

            response.setThrust( [ input_1[t],input_2[t],input_3[t],input_4[t] ])
            disturbed_response.setThrust( [ input_1[t]+disturbance[t],input_2[t]+disturbance[t],input_3[t]+disturbance[t],input_4[t]+disturbance[t] ])
            # temporary variables
            states = response.getAllStates()
            disturbed_states = disturbed_response.getAllStates()

############## UNDISTURBED #################################3

            P[t] = states[16]
            Q[t] = states[17]
            R[t] = states[18]

            Pdot[t] = states[13]
            Qdot[t] = states[14]
            Rdot[t] = states[15]

            U[t] = states[7]
            V[t] = states[8]
            W[t] = states[9]

            Udot[t] = states[10]
            Vdot[t] = states[11]
            Wdot[t] = states[12]

            L[t] = states[19]
            M[t] = states[20]
            N[t] = states[21]

            X[t] = states[22]
            Y[t] = states[23]
            Z[t] = states[24]

################### DISTURBED #############################################3

            P_disturb[t] = disturbed_states[16]
            Q_disturb[t] = disturbed_states[17]
            R_disturb[t] = disturbed_states[18]

            Pdot_disturb[t] = disturbed_states[13]
            Qdot_disturb[t] = disturbed_states[14]
            Rdot_disturb[t] = disturbed_states[15]

            U_disturb[t] = disturbed_states[7]
            V_disturb[t] = disturbed_states[8]
            W_disturb[t] = disturbed_states[9]

            Udot_disturb[t] = disturbed_states[10]
            Vdot_disturb[t] = disturbed_states[11]
            Wdot_disturb[t] = disturbed_states[12]

            L_disturb[t] = disturbed_states[19]
            M_disturb[t] = disturbed_states[20]
            N_disturb[t] = disturbed_states[21]

            X_disturb[t] = disturbed_states[22]
            Y_disturb[t] = disturbed_states[23]
            Z_disturb[t] = disturbed_states[24]

            # next time step
            simError = response.step()
            disturb_simError = disturbed_response.step()

            if(simError == 1 or disturb_simError == 1):
                t = timeSteps

            else:
                t += 1

        # Saves response in *.npz file
        # print(system)
        if(simError == 0 and disturb_simError == 0):
            np.savez(filename,input_1=input_1,input_2=input_2,input_3=input_3,\
                    input_4=input_4,P=P,Q=Q,R=R,U=U,V=V,W=W,Pdot=Pdot,Qdot=Qdot,\
                    Rdot=Rdot,Udot=Udot,Vdot=Vdot,Wdot=Wdot,system=drone_info,\
                    X=X,Y=Y,Z=Z,L=L,M=M,N=N,\
                    disturbance=disturbance,disturbance_labels=disturbance_labels,\
                    P_disturb=P_disturb,Q_disturb=Q_disturb,R_disturb=R_disturb,\
                    U_disturb=U_disturb,V_disturb=V_disturb,W_disturb=W_disturb,\
                    Pdot_disturb=Pdot_disturb,Qdot_disturb=Qdot_disturb,\
                    Rdot_disturb=Rdot_disturb,Udot_disturb=Udot_disturb,\
                    Vdot_disturb=Vdot_disturb,Wdot_disturb=Wdot_disturb,\
                    X_disturb=X_disturb,Y_disturb=Y_disturb,Z_disturb=Z_disturb,\
                    L_disturb=L_disturb,M_disturb=M_disturb,N_disturb=N_disturb\
                    )

            # Change number on filename to correspond to simulation number
            filename = filename.replace(str(i),str(i+1))
            numSim.update()
            i += 1
