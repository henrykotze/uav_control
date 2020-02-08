import numpy as np
import random
import h5py
import tensorflow as tf
from esl_timeseries_dataset import esl_timeseries_dataset



dataset_name='./test.h5f'

fake_dataset = np.arange(0,20).reshape(5,4)
print(fake_dataset)


h5f = h5py.File(str(dataset_name),'w')
h5f.create_dataset('dataset', data=fake_dataset)
h5f.close()

window_size=1
input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [11, 12, 13, 14, 15, 16]


test_dataset = esl_timeseries_dataset(dataset_name,window_size,1,1,input_indices,
                output_indices,shuffle=False)
print(test_dataset[1,:])


#
# print(dataset)
# dataset[0,:] = dataset[0,:]/10
# print(dataset)
#
# input_indices=[0,1,5]
# output_indices=[4,5]
#
# windowsize=2
# step=1
# batchsize=3
#
# meep = esl_timeseries_dataset(dataset,windowsize,step,batchsize,input_indices,output_indices,shuffle=False)
#
# for x_train,y_train in meep:
#
#     print(x_train)
#     print(y_train)
#     print('===============')
