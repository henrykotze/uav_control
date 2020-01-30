import numpy as np
import random
import h5py
import tensorflow as tf


class esl_timeseries_dataset(object):
    '''
    Holds entire N-dimensional timeseries dataset and iterates over
    the dataset without keeping copies of entries within the dataset, by using
    indices
    '''

    def __init__(self,dataset,windowsize,step,batchsize,input_indices,ouput_indices,shuffle=True):

        self.batchsize = batchsize
        self.step = step
        self.windowsize = windowsize
        self.n = 0
        self.x_indices = []
        self.y_indices = []
        self.shuffle_dataset = shuffle
        self.input_indices = input_indices
        self.output_indices = output_indices


        # self.load_dataset(dataset)
        self.dataset = dataset
        self.shape = self.dataset.shape
        self.total_samples = self.shape[1]
        self.total_features = len(self.input_indices)
        self.total_labels = len(self.output_indices)
        self.num_batches = int(np.ceil(self.total_samples/self.batchsize))

    def get_input_shape(self):
        return int(self.windowsize*self.total_features)

    def load_dataset(self,path_to_h5py):

        print('\n--------------------------------------------------------------')
        print('Reading dataset file: {}'.format(path_to_h5py))
        print('--------------------------------------------------------------')
        f = h5py.File(path_to_h5py, 'r')
        # print('{} contains: {}'.format(path_to_h5py,f.keys()))
        self.dataset = f['dataset']
        self.shape = self.dataset.shape
        self.total_samples = self.shape[1]
        self.total_features = len(self.input_indices)
        self.total_labels = len(self.output_indices)
        self.num_batches = int(np.ceil(self.total_samples/self.batchsize))

    def __iter__(self):

        start_index = self.windowsize

        for i in range(start_index, self.total_samples):
            indices = range(i-self.windowsize, i, self.step)
            (self.x_indices).append(indices)
            (self.y_indices).append(i)

        self.num_indices = len(self.x_indices)

        if(self.shuffle_dataset):
            self.shuffle()

        return self

    def __next__(self):

        x_train = np.zeros((self.batchsize,self.windowsize*self.total_features))
        y_train = np.zeros((self.batchsize,self.total_labels))

        if(self.n < self.num_batches):

            for batch_counter in range(self.batchsize):

                x_train[batch_counter,:] = self.dataset[:,self.x_indices[self.n+batch_counter]][:,self.input_indices].flatten()
                y_train[batch_counter,:] = self.dataset[:,self.y_indices[self.n+batch_counter]][:,self.output_indices].flatten()


            self.n += self.batchsize
            return x_train,y_train

        else:
            self.n = 0
            raise StopIteration


    def shuffle(self):
        c = list(zip(self.x_indices,self.y_indices))
        random.shuffle(c)
        self.x_indices, self.y_indices = zip(*c)
        self.x_indices = list(self.x_indices)
        self.y_indices = list(self.y_indices)
        # print(self.x_indices)





x1 = np.arange(0,10,1)
x2 = np.arange(10,20,1)
x3 = np.arange(20,30,1)

dataset = np.zeros((3,10))
dataset[0,:] = x1
dataset[1,:] = x2
dataset[2,:] = x3

input_indices=[0,1,2,3]
output_indices=[4,5,6,7,9]

windowsize=3
step=1
batchsize=3

meep = esl_timeseries_dataset(dataset,windowsize,step,batchsize,input_indices,output_indices,shuffle=False)

for x_train,y_train in meep:

    print(x_train)
    print(y_train)
    print('===============')
