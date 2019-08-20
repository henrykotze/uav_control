#!/usr/bin/env python3
# from uav_model import drone
import numpy as np
# import matplotlib as mpl
# # mpl.use('tkagg')
# # mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import argparse
import shelve

parser = argparse.ArgumentParser(\
        prog='Plot Data',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-n', default=0, help='plot a specific number response')


args = parser.parse_args()

dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
number = int(vars(args)['n'])




print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/readme'))
print('----------------------------------------------------------------')
with shelve.open( str(dir+'/readme')) as db:
    t = int(db['t'])
    dt = float(db['dt'])
    numberSims = int(db['numSim'])
    max_input = float(db['maxInput'])


    for key,value in db.items():
        print("{}: {}".format(key, value))
db.close()







div = 10
timeSteps = int((t/dt))

features = np.zeros( int(timeSteps/div) )   # +1 is for the input


filename = filename.replace(str(0),str(number))
print("Plotting Response of file: ", filename)


data = np.load(filename)

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
i = 0

data.close()



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)


# plt.figure(10)
# plt.title('Torque by motors')
# plt.plot(features,'.-', mew=1, ms=8,mec='w')
# plt.plot(Pdot,'-', mew=1, ms=8,mec='w')

plt.figure(1)
plt.title('Thrust Delivered By Motors')
plt.plot(input_1,'-', mew=1, ms=8,mec='w')
# plt.plot(input_2,'-', mew=1, ms=8,mec='w')
# plt.plot(input_3,'-', mew=1, ms=8,mec='w')
# plt.plot(input_4,'-', mew=1, ms=8,mec='w')
# plt.legend(['$T_{1}$','$T_{2}$','$T_{3}$','$T_{4}$',])
plt.xlabel("Time -[$\mu$s]")
plt.ylabel("Thrust - [N]")
plt.grid()


plt.show()

# plt.figure(2)
# plt.plot(P,'-', mew=1, ms=8,mec='w')
# plt.plot(Q,'-', mew=1, ms=8,mec='w')
# plt.plot(R,'-', mew=1, ms=8,mec='w')
# plt.grid()
#
#
# plt.figure(3)
# plt.title('Velocity')
# plt.plot(V,'-', mew=1, ms=8,mec='w')
# plt.plot(W,'-', mew=1, ms=8,mec='w')
# plt.plot(U,'-', mew=1, ms=8,mec='w')
# plt.grid()
#
#
# plt.figure(4)
# plt.title('Acceleration In X,Y,Z Directions')
# plt.plot(Vdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Wdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Udot,'-', mew=1, ms=8,mec='w')
# plt.legend(['$\dot{V}$','$\dot{W}$','$\dot{U}$'])
# plt.xlabel("Time -[$\mu$s]")
# plt.ylabel('Acceleration - [m/s$^{2}$]')
# plt.grid()
#
#
# plt.figure(5)
# plt.title('Angular Acceleration of Drone')
# plt.plot(Pdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Qdot,'-', mew=1, ms=8,mec='w')
# plt.plot(Rdot,'-', mew=1, ms=8,mec='w')
# plt.legend(['$\dot{P}$','$\dot{Q}$','$\dot{R}$'])
# plt.xlabel("Time -[$\mu$s]")
# plt.ylabel('Angular Acceleration - [rad/s$^{2}$]')
# plt.grid()
#
#
#
#
#
#
# plt.show()
