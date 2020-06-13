

import matplotlib.pyplot as plt
import numpy as np
import h5py


path_to_h5py = "./dataset_sim/test_disturbance"


print('\n--------------------------------------------------------------')
print('Reading dataset file: {}'.format(path_to_h5py))
print('--------------------------------------------------------------')
hf = h5py.File(path_to_h5py, 'r+')
# print('{} contains: {}'.format(path_to_h5py,f.keys()))
dataset = hf['dataset'][:]
shape = dataset.shape
total_samples = int(np.floor(shape[1]))

hf.close()


meep1 = dataset[0,:]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.title('X: Velocity and Disturbance')
plt.plot(dataset[8,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[11,:],'-', mew=1, ms=8,mec='w')
plt.grid()
# plt.plot(dataset[5,:],'-', mew=1, ms=8,mec='w')
# plt.plot(dataset[6,:],'-', mew=1, ms=8,mec='w')



plt.figure(2)
plt.title('Y: Velocity and Disturbance')
plt.plot(dataset[9,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[12,:],'-', mew=1, ms=8,mec='w')
plt.grid()
#
#
#
plt.figure(3)
plt.title('Z: Velocity and Disturbance')
plt.plot(dataset[10,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[13,:],'-', mew=1, ms=8,mec='w')
plt.grid()




plt.figure(4)
plt.plot(dataset[19,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[22,:],'-', mew=1, ms=8,mec='w')
plt.grid()
# plt.plot(dataset[5,:],'-', mew=1, ms=8,mec='w')
# plt.plot(dataset[6,:],'-', mew=1, ms=8,mec='w')



plt.figure(5)
plt.plot(dataset[20,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[23,:],'-', mew=1, ms=8,mec='w')
plt.grid()
#
#
#
plt.figure(6)
plt.plot(dataset[21,:],'-', mew=1, ms=8,mec='w')
plt.plot(dataset[24,:],'-', mew=1, ms=8,mec='w')
plt.grid()
#
# # plt.title('Pitch Rate')
# # plt.xlabel('Time - [s]')
# # plt.ylabel('Pitch Rate - [rad/s]')



plt.show()
