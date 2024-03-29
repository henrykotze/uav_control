import numpy as np
import tensorflow as tf




x1 = np.arange(0,10,1)
x2 = np.arange(10,20,1)
x3 = np.arange(20,30,1)

dataset = np.zeros((3,10))
dataset[0,:] = x1
dataset[1,:] = x2
dataset[2,:] = x3
# print(dataset)


# Needs some cleaning
def load_bad_tf_dataset(path2h5py,windowsize,train_split):

    print('\n--------------------------------------------------------------')
    print('Reading dataset file: {}'.format(path_to_h5py))
    print('--------------------------------------------------------------')
    f = h5py.File(path_to_h5py, 'r')
    # print('{} contains: {}'.format(path_to_h5py,f.keys()))
    dataset = f['dataset']


    past_history = windowsize # Time window
    future_target = 0  # How far in the future does the nn predict
    STEP = 1 # Step between samples
    TRAIN_SPLIT=train_split #

    x_train, y_train = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    return train_data_single



def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        # print(dataset[:,indices].flatten())
        data.append(dataset[:,indices].flatten())

        if single_step:
            labels.append(dataset[:,i+target_size].flatten())
            # print(labels)
            # labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

print("--------------------")
print("--------------------")
print("--------------------")


past_history = 1
future_target = 0
STEP = 1
TRAIN_SPLIT=9

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().batch(2)



for x,y in train_data_single:
    print(x)

# print('Single window of past history : {}'.format(x_train_single[0].shape))

print(x_train_single)
print("------------")
print(y_train_single)
