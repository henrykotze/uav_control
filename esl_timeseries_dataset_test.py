import numpy as np
import random
import h5py
import tensorflow as tf
from esl_timeseries_dataset import esl_timeseries_dataset



dataset_name='./test.h5f'

fake_dataset = np.arange(0,80).reshape(8,10)
print(fake_dataset)


h5f = h5py.File(str(dataset_name),'w')
h5f.create_dataset('dataset', data=fake_dataset)
h5f.close()

window_size=2
input_indices= [0, 1, 2, 4]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [5,6,7]

batchsize=4
test_dataset = esl_timeseries_dataset(dataset_name,window_size,1,batchsize,input_indices,
                output_indices,shuffle=False)





for x_train,y_train in test_dataset:

    print(x_train)
    print(y_train)
    print('===============')
