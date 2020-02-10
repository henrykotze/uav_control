import numpy as np
import random
import h5py
from esl_timeseries_dataset import esl_timeseries_dataset



dataset_name='./test.h5f'

fake_dataset = np.arange(0,1000).reshape(10,100)
print(fake_dataset)


h5f = h5py.File(str(dataset_name),'w')
h5f.create_dataset('dataset', data=fake_dataset)
h5f.close()

window_size=10
input_indices= [0, 1, 2, 4]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [7,8,9]

batchsize=10
test_dataset = esl_timeseries_dataset(dataset_name,window_size,1,batchsize,input_indices,
                output_indices,shuffle=False)


print(test_dataset.getNumSamples())


c = 0 

for x_train,y_train in test_dataset:
#
     print(x_train)
     print(y_train)
     print("============")
     c+=1


print(test_dataset.getTotalPredictions())
print("total batches: {}".format(test_dataset.getTotalBatches()))
print("counter: {}".format(c))


