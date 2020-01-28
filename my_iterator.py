import numpy as np



class beter_iterator(object):

    def __init__(self,length,shape,windowsize,step,batchsize):

        self.batchsize = batchsize
        self.step = step
        self.windowsize = windowsize
        self.num_indices = 0
        self.n = int(np.ceil(self.length/self.windowsize))
        self.
        self.length = length

        self.x_indices = []
        self.y_indices = []

    def __iter__(self):
        start_index = self.windowsize

        for i in range(start_index, self.length):

            indices = range(i-self.windowsize, i, self.step)
            (self.x_indices).append(indices)
            (self.y_indices).append(i)

        self.num_indices = len(self.x_indices)

        return self

    def __next__(self):
        # Need to work in batches
        x_train = []
        y_train = []
        self.n += 1
        if(self.n < self.num_indices):
            for batch in range(self.batchsize):
                # meep.append([self.x_indices[self.n],self.y_indices[self.n]])
                x_train.append(self.x_indices[self.n])
                y_train.append(self.y_indices[self.n])
            # return [self.x_indices[self.n],self.y_indices[self.n]]
            return x_train,y_train
        else:
            raise StopIteration



    def shuffle(self):
        pass

x1 = np.arange(0,10,1)
x2 = np.arange(10,20,1)
x3 = np.arange(20,30,1)

dataset = np.zeros((3,10))
dataset[0,:] = x1
dataset[1,:] = x2
dataset[2,:] = x3

length = 10
shape = 3
windowsize=5
step=1
batchsize=4

meep = beter_iterator(length,shape,windowsize,step,batchsize)

for [x_train, y_train] in meep:
    # print(dataset[:,x_train])
    print(x_train)
    # print(dataset[:,y_train].flatten())
