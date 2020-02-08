import numpy as np
import random
import h5py
import tensorflow as tf


class esl_timeseries_dataset(object):
    '''
    Holds entire N-dimensional timeseries dataset and iterates more efficient
    over the dataset
    '''

    def __init__(self,dataset,windowsize,step,batchsize,input_indices,output_indices,shuffle=True,percent=1):

        self.batchsize = batchsize
        self.step = step
        self.windowsize = windowsize
        self.n = 0
        self.x_indices = []
        self.y_indices = []
        self.shuffle_dataset = shuffle
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.percent = percent


        self.load_dataset(dataset)
        # self.dataset = dataset
        # self.shape = self.dataset.shape
        # self.total_samples = self.shape[1]
        # self.total_inputs = len(self.input_indices)
        # self.total_labels = len(self.output_indices)
        # self.num_batches = int(np.ceil(self.total_samples/self.batchsize))

    def get_input_shape(self):
        return int(self.windowsize*self.total_inputs)

    def load_dataset(self,path_to_h5py):

        print('\n--------------------------------------------------------------')
        print('Reading dataset file: {}'.format(path_to_h5py))
        print('--------------------------------------------------------------')
        hf = h5py.File(path_to_h5py, 'r')
        # print('{} contains: {}'.format(path_to_h5py,f.keys()))
        self.dataset = hf['dataset']
        #hf.close()
        self.shape = self.dataset.shape
        self.total_samples = int(np.floor(self.shape[1]*self.percent))
        self.total_inputs = len(self.input_indices)
        self.total_labels = len(self.output_indices)
        self.num_batches = int(np.ceil(self.total_samples/self.batchsize))


        start_index = self.windowsize

        for i in range(start_index, self.total_samples):
            indices = range(i-self.windowsize, i, self.step)
            (self.x_indices).append(indices)
            (self.y_indices).append(i)

        self.num_indices = len(self.x_indices)

        if(self.shuffle_dataset):
            self.shuffle()

    def __iter__(self):
        return self

    def __next__(self):

        x_train = np.zeros((self.batchsize,self.windowsize*self.total_inputs))
        y_train = np.zeros((self.batchsize,self.total_labels))

        if(self.n < self.num_batches):

            for batch_counter in range(self.batchsize):

                x_train[batch_counter,:] = self.dataset[:,self.x_indices[self.n+batch_counter]][self.input_indices].flatten()
                y_train[batch_counter,:] = self.dataset[:,self.y_indices[self.n+batch_counter]][self.output_indices]


            self.n += self.batchsize
            return x_train,y_train

        else:
            self.n = 0
            raise StopIteration


    def __getitem__(self,key):
        if isinstance(key, tuple):
            col,row = key
            if isinstance(row, slice):
                return self.dataset[col,row]
            else:
                return self.dataset[col,row]




    def shuffle(self):
        c = list(zip(self.x_indices,self.y_indices))
        random.shuffle(c)
        self.x_indices, self.y_indices = zip(*c)
        self.x_indices = list(self.x_indices)
        self.y_indices = list(self.y_indices)
        # print(self.x_indices)
