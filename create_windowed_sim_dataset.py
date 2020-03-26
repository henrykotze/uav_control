import numpy as np
import h5py
import argparse
from tqdm import trange, tqdm
import shelve
from terminaltables import AsciiTable




parser = argparse.ArgumentParser(\
        prog='Create datasets from simulated data',\
        description=''
        )


parser.add_argument('-dataset_path', default='', help='path to training dataset')
parser.add_argument('-Nt', default=10, help='window size')
parser.add_argument('-step', default=1, help='step size between samples')
parser.add_argument('-saveTo', default=1, help='save windowed dataset to:')
parser.add_argument('-windowed_dataset_name', default=1, help='name of windowed dataset')


args = parser.parse_args()

dataset_path = str(vars(args)['dataset_path'])
windowed_dataset_name = str(vars(args)['windowed_dataset_name'])
windowed_dataset_loc = str(vars(args)['saveTo'])
window_size = int(vars(args)['Nt'])
step = int(vars(args)['step'])
dataset_readme = dataset_path+'_readme'
windowed_dataset_readme = windowed_dataset_name+'_readme'


print('----------------------------------------------------------------')
print('Fetching dataset info from: {}'.format(dataset_readme))
print('----------------------------------------------------------------')
data = []

with shelve.open(dataset_readme) as db:
    for key,value in db.items():
        data.append([str(key),str(value)])
db.close()
table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)



print('\n--------------------------------------------------------------')
print('Saving windowed dataset readme:', str(windowed_dataset_loc + '/'+ windowed_dataset_readme))
print('--------------------------------------------------------------')

with shelve.open( str(windowed_dataset_loc + '/'+ windowed_dataset_readme)) as db:

    with shelve.open(dataset_readme) as db2:
        name_validation_dataset = db2['name_of_validation_dataset']
        dataset_loc = db2['dataset_loc']

        for key,value in db2.items():
            db[str(key)] = value

    db2.close()

    db['window_size'] = window_size
    db['windowed_dataset_name'] = windowed_dataset_name
    db['windowed_dataset_loc'] = windowed_dataset_loc
    db['step'] = step

db.close()


def mem_ineff_dataset(dataset_name,windowsize,step,input_indices,output_indices,shuffle=True):

    n = 0
    x_indices = []
    y_indices = []

    shuffle_dataset = shuffle

    print('\n--------------------------------------------------------------')
    print('Reading dataset file: {}'.format(dataset_name))
    print('--------------------------------------------------------------')
    hf = h5py.File(dataset_name, 'r+')
    # print('{} contains: {}'.format(path_to_h5py,f.keys()))
    dataset = hf['dataset']
    # close()
    shape = dataset.shape
    total_samples = int(np.floor(shape[1]))
    total_inputs = len(input_indices)
    total_labels = len(output_indices)

    start_index = windowsize

    features = np.zeros((total_samples-windowsize,len(input_indices)*window_size))
    labels = np.zeros((total_samples-windowsize,len(output_indices)))

    range_progressbar = trange(start_index, total_samples, desc='sample #', leave=True)

    for i in range_progressbar:
        range_progressbar.set_description("sample #")
        range_progressbar.refresh() # to show immediately the update
        range_progressbar.update()

        indices = range(i-windowsize, i, step)

        features[i-windowsize,:] = dataset[:,indices][input_indices].flatten()
        labels[i-windowsize,:] = dataset[:,i][output_indices]

    hf.close()
    return features,labels




# [q1,q2,q3,q4,U,V,W,T1,T2,T3,T4]
input_indices= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [P,Q,R,Udot,Vdot,Wdot]
output_indices = [11, 12, 13, 14, 15, 16]
x,y = mem_ineff_dataset(dataset_path,window_size,1,input_indices,
                output_indices,shuffle=False)


print('\n--------------------------------------------------------------')
print('Saving windowed dataset:',windowed_dataset_name)
print('--------------------------------------------------------------')

h5f = h5py.File(str(windowed_dataset_loc + '/'+ windowed_dataset_name),'w')
h5f.create_dataset('features', data=x)
h5f.create_dataset('labels', data=y)
h5f.close()
