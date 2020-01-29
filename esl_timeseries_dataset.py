import numpy as np
import random


class esl_timeseries_dataset(object):
    '''
    Holds entire N-dimensional timeseries dataset and iterates over
    the dataset without keeping copies of entries within the dataset, by using
    indices
    '''

    def __init__(self,dataset,windowsize,step,batchsizes,shuffle=True):

        self.batchsize = batchsize
        self.step = step
        self.windowsize = windowsize
        self.n = 0
        self.x_indices = []
        self.y_indices = []
        self.shuffle_dataset = shuffle


        self.load_dataset(dataset)
        # self.dataset = dataset
        # self.shape = self.dataset.shape
        # self.total_samples = self.shape[1]
        # self.total_features = self.shape[0]
        # self.num_batches = int(np.ceil(self.total_samples/self.batchsize))



    def load_dataset(self,path_to_h5py):

        print('\n--------------------------------------------------------------')
        print('Reading dataset file: {}'.format(path_to_h5py))
        print('--------------------------------------------------------------')
        f = h5py.File(path_to_h5py, 'r')
        # print('{} contains: {}'.format(path_to_h5py,f.keys()))
        self.dataset = f['dataset']
        self.shape = self.dataset.shape
        self.total_samples = self.shape[1]
        self.total_features = self.shape[0]
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
        y_train = np.zeros((self.batchsize,self.total_features))

        if(self.n < self.num_batches):

            for batch_counter in range(self.batchsize):

                x_train[batch_counter,:] = self.dataset[:,self.x_indices[self.n+batch_counter]].flatten()
                y_train[batch_counter,:] = self.dataset[:,self.y_indices[self.n+batch_counter]].flatten()


            self.n += self.batchsize
            return x_train,y_train

        else:
            raise StopIteration


    def shuffle(self):
        c = list(zip(self.x_indices,self.y_indices))
        random.shuffle(c)
        self.x_indices, self.y_indices = zip(*c)





# x1 = np.arange(0,10,1)
# x2 = np.arange(10,20,1)
# x3 = np.arange(20,30,1)
#
# dataset = np.zeros((3,10))
# dataset[0,:] = x1
# dataset[1,:] = x2
# dataset[2,:] = x3
#
# windowsize=3
# step=1
# batchsize=3
#
# meep = esl_timeseries_dataset(dataset,windowsize,step,batchsize)
#
# for x_train,y_train in meep:
#
#     print(x_train)
#     print(y_train)
#     print('===============')
