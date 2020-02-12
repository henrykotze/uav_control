import numpy as np
import ulog2csv
import os
from core import ULog
import pandas as pd
import math
from tqdm import trange, tqdm
import argparse
import shelve
import h5py
from terminaltables import AsciiTable
import yaml



parser = argparse.ArgumentParser(\
        prog='Generates h5py dataset from PX4 Ulog files',\
        description='Creates .npz files containing the responses of a rotary wing UAV'
        )


parser.add_argument('-logdir', default='./logs/', help='path to the logs parent direcory')
parser.add_argument('-drone', default='', help='path to the file containing drone corresponding to logs')
parser.add_argument('-add_info', default='', help='add additional information')
parser.add_argument('-dataset_name', default='dataset', help='name of saved dataset')
parser.add_argument('-dataset_loc', default='./', help='path to where dataset is saved')
parser.add_argument('-val_percent', default=0.1, help='percentage of logs to be used for validations')

args = parser.parse_args()



logdir = vars(args)['logdir']
drone_file_path = str(vars(args)['drone'])
addition_info = str(vars(args)['add_info'])
dataset_name = str(vars(args)['dataset_name'])
validation_dataset_name = 'validation_' + str(vars(args)['dataset_name'])
dataset_loc = str(vars(args)['dataset_loc'])
validation_percentage = float(vars(args)['val_percent'])

if(drone_file_path == ''):
    raise Exception("Provide a drone file")


yamlfile = open(drone_file_path,'r')
drone_info = yaml.load(yamlfile,Loader=yaml.Loader)
drone_name = str(drone_info['drone']['name'])
motor_thrust = float(drone_info['drone']['motor_thrust'])


if(not os.path.isdir(dataset_loc)):
    raise NameError('Path to store dataset: {}, does not exist'.format(dataset_loc))



def convertInertia2Body(q,inertia):

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]


    r11 = q0**2 + q1**2 + q2**2 +q3**2
    r12 = 2*(q1*q2 +q0*q3)
    r13 = 2*(q1*q3 - q0*q2)

    r21 = 2*(q1*q2-q0*q3)
    r22 = q0**2 - q1**2 + q2**2 - q3**2
    r23 = 2*(q2*q3 + q0*q1)

    r31 = 2*(q1*q3 + q0*q2)
    r32 = 2*(q2*q3 - q0*q1)
    r33 = q0**2 - q1**2 - q2**2 + q3**2

    # print(inertia[0].size)

    U = r11*inertia[0] + r12*inertia[1] + r13*inertia[2]
    V = r21*inertia[0] + r22*inertia[1] + r23*inertia[2]
    W = r31*inertia[0] + r32*inertia[1] + r33*inertia[2]

    return [U,V,W]


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        # return array[idx-1]
        return idx-1
    else:
        # return array[idx]
        return idx

def convert_ulog2csv(ulog_file_name, messages, output, delimiter, disable_str_exceptions=False):
    """
    Coverts and ULog file to a CSV file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """

    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    output_file_prefix = ulog_file_name
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]

    # write to different output path?
    if output:
        base_name = os.path.basename(output_file_prefix)
        dir_name = os.path.dirname(output_file_prefix).split('/')[-1]
        base_name = dir_name + '_' + base_name

        output_file_prefix = os.path.join(output, base_name)

    path2CSVs = []


    # print(output_file_prefix)
    for d in data:
        fmt = '{0}_{1}_{2}.csv'
        output_file_name = fmt.format(output_file_prefix, d.name, d.multi_id)
        fmt = 'Writing {0} ({1} data points)'
        # print(fmt.format(output_file_name, len(d.data['timestamp'])))
        path2CSVs.append(output_file_name)

        with open(output_file_name, 'w') as csvfile:

            # use same field order as in the log, except for the timestamp
            data_keys = [f.field_name for f in d.field_data]
            data_keys.remove('timestamp')
            data_keys.insert(0, 'timestamp')  # we want timestamp at first position

            # we don't use np.savetxt, because we have multiple arrays with
            # potentially different data types. However the following is quite
            # slow...

            # write the header
            csvfile.write(delimiter.join(data_keys) + '\n')

            # write the data
            last_elem = len(data_keys)-1
            for i in range(len(d.data['timestamp'])):
                for k in range(len(data_keys)):
                    csvfile.write(str(d.data[data_keys[k]][i]))
                    if k != last_elem:
                        csvfile.write(delimiter)
                csvfile.write('\n')


    return path2CSVs


def whitening_dataset(dataset,features):
    ''' Whitening of dataset '''
    mean = np.mean(dataset,1).reshape(features,1)
    std = np.std(dataset,1).reshape(features,1)

    dataset = (dataset-mean)/std
    return [dataset,mean,std]


def normalise_dataset(dataset,features):
    max = np.amax(dataset,1)

def generateDataset(logs):

    features = 17
    dataset_num_entries=10000000
    counter = 0
    dataset = np.zeros((features,dataset_num_entries))
    ulog_entry_prev = ''

    for ulog_entry in tqdm(logs, desc = "log file: {}".format(str(ulog_entry_prev))):
        ulog_entry_prev = ulog_entry
        print(ulog_entry)

        listOfCSVs = convert_ulog2csv(ulog_entry,log_eoi,'./',',')

        # print(listOfCSVs)

        # print([s for s in listOfCSVs if listOfEOI[0] in s][0])

        # vehicle_attitude csv
        df = pd.read_csv([s for s in listOfCSVs if listOfEOI[1] in s][0])

        timestamp_attitude = np.array(df['timestamp'].tolist())
        P = np.array(df['rollspeed'].tolist()).astype(float)
        Q = np.array(df['pitchspeed'].tolist()).astype(float)
        R = np.array(df['yawspeed'].tolist()).astype(float)
        q1 = np.array(df['q[0]'].tolist()).astype(float)
        q2 = np.array(df['q[1]'].tolist()).astype(float)
        q3 = np.array(df['q[2]'].tolist()).astype(float)
        q4 = np.array(df['q[3]'].tolist()).astype(float)
        q = [q1,q2,q3,q4]
        length_attitude = timestamp_attitude.size



        # vechile_local_position csv
        df = pd.read_csv([s for s in listOfCSVs if listOfEOI[0] in s][0])
        timestamp_local_position = np.array(df['timestamp'].tolist())
        inertia_U = np.array(df['vx'].tolist()).astype(float)
        inertia_V = np.array(df['vy'].tolist()).astype(float)
        inertia_W = np.array(df['vz'].tolist()).astype(float)
        inertia_Udot = np.array(df['ax'].tolist()).astype(float)
        inertia_Vdot = np.array(df['ay'].tolist()).astype(float)
        inertia_Wdot = np.array(df['az'].tolist()).astype(float)
        size_local_position = timestamp_local_position.size


        # vechile_actuator_outputs csv
        df = pd.read_csv([s for s in listOfCSVs if listOfEOI[2] in s][0])

        timestamp_actuator_outputs = np.array(df['timestamp'].tolist())
        timestamp_actuator_outputs_cor = np.zeros(length_attitude)
        PWM_0 = np.array(df['output[0]'].tolist()).astype(float)
        PWM_1 = np.array(df['output[1]'].tolist()).astype(float)
        PWM_2 = np.array(df['output[2]'].tolist()).astype(float)
        PWM_3 = np.array(df['output[3]'].tolist()).astype(float)

        # not all entries are written at the same time in the log file, thus we
        # will align them all by assign them to closet entry that corresponds in the
        # local_position log
        P_ = np.zeros(len(timestamp_local_position),dtype=float)
        Q_ = np.zeros(len(timestamp_local_position),dtype=float)
        R_ = np.zeros(len(timestamp_local_position),dtype=float)
        q1_ = np.zeros(len(timestamp_local_position),dtype=float)
        q2_ = np.zeros(len(timestamp_local_position),dtype=float)
        q3_ = np.zeros(len(timestamp_local_position),dtype=float)
        q4_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_0_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_1_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_2_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_3_ = np.zeros(len(timestamp_local_position),dtype=float)

        time_keeper = 0

        for value in timestamp_local_position:
            loc = find_nearest(timestamp_actuator_outputs,value)
            PWM_0_[time_keeper] = PWM_0[loc]
            PWM_1_[time_keeper] = PWM_1[loc]
            PWM_2_[time_keeper] = PWM_2[loc]
            PWM_3_[time_keeper] = PWM_3[loc]

            time_keeper += 1


        time_keeper = 0
        for value in timestamp_local_position:
            loc = find_nearest(timestamp_attitude,value)
            P_[time_keeper] = P[loc]
            Q_[time_keeper] = Q[loc]
            R_[time_keeper] = R[loc]
            q1_[time_keeper] = q1[loc]
            q2_[time_keeper] = q2[loc]
            q3_[time_keeper] = q3[loc]
            q4_[time_keeper] = q4[loc]

            time_keeper += 1

        q_ = [q1_,q2_,q3_,q4_]


        inertia_Vel = [inertia_U,inertia_V,inertia_W]
        inertia_Acc = [inertia_Udot,inertia_Vdot,inertia_Wdot]

        [U_,V_,W_] = convertInertia2Body(q_,inertia_Vel)
        [Udot,Vdot,Wdot] = convertInertia2Body(q_,inertia_Acc)

        T1 = (PWM_0_/1000 - 1)*motor_thrust*9.81
        T2 = (PWM_1_/1000 - 1)*motor_thrust*9.81
        T3 = (PWM_2_/1000 - 1)*motor_thrust*9.81
        T4 = (PWM_3_/1000 - 1)*motor_thrust*9.81


        dataset[0,counter:counter  + size_local_position] = q1_
        dataset[1,counter:counter  + size_local_position] = q2_
        dataset[2,counter:counter  + size_local_position] = q3_
        dataset[3,counter:counter  + size_local_position] = q4_
        dataset[4,counter:counter  + size_local_position] = U_
        dataset[5,counter:counter  + size_local_position] = V_
        dataset[6,counter:counter  + size_local_position] = W_
        dataset[7,counter:counter  + size_local_position] = T1
        dataset[8,counter:counter  + size_local_position] = T2
        dataset[9,counter:counter  + size_local_position] = T3
        dataset[10,counter:counter + size_local_position] = T4
        dataset[11,counter:counter + size_local_position] = P_
        dataset[12,counter:counter + size_local_position] = Q_
        dataset[13,counter:counter + size_local_position] = R_
        dataset[14,counter:counter + size_local_position] = Udot
        dataset[15,counter:counter + size_local_position] = Vdot
        dataset[16,counter:counter + size_local_position] = Wdot


        for csvFile in listOfCSVs:
            os.remove(csvFile)

        # perhaps delete *.csv
        counter += size_local_position

        if(counter >= dataset_num_entries):
            dataset_num_entries += 1000000
            dataset = np.hstack((dataset,np.zeros(features,1000000)))

    # resize dataset by using counter variable
    dataset = np.delete(dataset,slice(counter,dataset_num_entries,1),1)
    return dataset

# log directory
log_dir = "/home/henry/esl-sun/PX4/build/px4_sitl_default/logs/"
# log_dir = "./logs/"
# entries of interest
log_eoi = 'vehicle_local_position,vehicle_attitude,actuator_outputs'
listOfEOI = log_eoi.split(',')


description=''




numOfLogs=0
numOfValLogs=0
listOfLogs=[]

validation_logs=[]
training_logs=[]


# find all *.ulog files in log directory
for root, dirs, files in os.walk(log_dir):
    for name in files:
        numOfLogs = numOfLogs + 1
        log_name = str(os.path.join(root,name))
        listOfLogs.append(log_name)

# convert all found *.ulog files into *.cvs type format and store in desired dir
# with messages of interest
# for ulog_entru in tqdm(listOfLogs):
#     print(ulog_entru)

numOfValLogs = int(np.floor(numOfLogs*validation_percentage))

for x in range(numOfValLogs):
    lognum = np.random.randint(0,numOfLogs)
    validation_logs.append(listOfLogs[lognum])

training_logs = [e for e in listOfLogs if e not in validation_logs]


train_dataset = generateDataset(training_logs)
validation_dataset = generateDataset(validation_logs)


train_num_samples=len(train_dataset[0,:])
val_num_samples=len(validation_dataset[0,:])


max_q1 = np.amax(train_dataset[0,:])
max_q2 = np.amax(train_dataset[1,:])
max_q3 = np.amax(train_dataset[2,:])
max_q4 = np.amax(train_dataset[3,:])

maxU = np.amax(train_dataset[4,:])
maxV = np.amax(train_dataset[5,:])
maxW = np.amax(train_dataset[6,:])

max_T1 = np.amax(train_dataset[7,:])
max_T2 = np.amax(train_dataset[8,:])
max_T3 = np.amax(train_dataset[9,:])
max_T4 = np.amax(train_dataset[10,:])

maxP = np.amax(train_dataset[11,:])
maxQ = np.amax(train_dataset[12,:])
maxR = np.amax(train_dataset[13,:])

maxUdot = np.amax(train_dataset[14,:])
maxVdot = np.amax(train_dataset[15,:])
maxWdot = np.amax(train_dataset[16,:])


train_dataset[0,:] = train_dataset[0,:]
train_dataset[1,:] = train_dataset[1,:]
train_dataset[2,:] = train_dataset[2,:]
train_dataset[3,:] = train_dataset[3,:]
train_dataset[4,:] = train_dataset[4,:]/maxU
train_dataset[5,:] = train_dataset[5,:]/maxV
train_dataset[6,:] = train_dataset[6,:]/maxW
train_dataset[7,:] = train_dataset[7,:]/max_T1
train_dataset[8,:] = train_dataset[8,:]/max_T2
train_dataset[9,:] = train_dataset[9,:]/max_T3
train_dataset[10,:] = train_dataset[10,:]/max_T4
train_dataset[11,:] = train_dataset[11,:]/maxP
train_dataset[12,:] = train_dataset[12,:]/maxQ
train_dataset[13,:] = train_dataset[13,:]/maxR
train_dataset[14,:] = train_dataset[14,:]/maxUdot
train_dataset[15,:] = train_dataset[15,:]/maxVdot
train_dataset[16,:] = train_dataset[16,:]/maxWdot


validation_dataset[0,:] = validation_dataset[0,:]
validation_dataset[1,:] = validation_dataset[1,:]
validation_dataset[2,:] = validation_dataset[2,:]
validation_dataset[3,:] = validation_dataset[3,:]
validation_dataset[4,:] = validation_dataset[4,:]/maxU
validation_dataset[5,:] = validation_dataset[5,:]/maxV
validation_dataset[6,:] = validation_dataset[6,:]/maxW
validation_dataset[7,:] = validation_dataset[7,:]/max_T1
validation_dataset[8,:] = validation_dataset[8,:]/max_T2
validation_dataset[9,:] = validation_dataset[9,:]/max_T3
validation_dataset[10,:] = validation_dataset[10,:]/max_T4
validation_dataset[11,:] = validation_dataset[11,:]/maxP
validation_dataset[12,:] = validation_dataset[12,:]/maxQ
validation_dataset[13,:] = validation_dataset[13,:]/maxR
validation_dataset[14,:] = validation_dataset[14,:]/maxUdot
validation_dataset[15,:] = validation_dataset[15,:]/maxVdot
validation_dataset[16,:] = validation_dataset[16,:]/maxWdot


print('\n--------------------------------------------------------------')
print('Saving dataset readme at:', str(dataset_loc + '/'+dataset_name+'_readme'))
print('--------------------------------------------------------------')
with shelve.open( str(dataset_loc + '/'+dataset_name+'_readme')) as db:

    db['maxUdot'] = maxUdot
    db['maxVdot'] = maxVdot
    db['maxWdot'] = maxWdot

    db['maxU'] = maxU
    db['maxV'] = maxV
    db['maxW'] = maxW

    db['maxP'] = maxP
    db['maxQ'] = maxQ
    db['maxR'] = maxR

    db['max_T1'] = max_T1
    db['max_T2'] = max_T2
    db['max_T3'] = max_T3
    db['max_T4'] = max_T4

    db['max_q1'] = max_q1
    db['max_q2'] = max_q2
    db['max_q3'] = max_q3
    db['max_q4'] = max_q4

    db['train_dataset_num_entries'] = train_num_samples
    db['validation_dataset_num_entries'] = val_num_samples
    db['numOfLogs'] = len(listOfLogs)
    db['drone_name'] = drone_name
    db['motor_thrust'] = motor_thrust
    db['addition_information'] = str(addition_info)
    db['name_of_validation_dataset'] = validation_dataset_name
    db['validation_percentage'] =  validation_percentage
    db['dataset_loc'] = dataset_loc



db.close()


print('\n--------------------------------------------------------------')
print('Saving training dataset:', dataset_name)
print('--------------------------------------------------------------')

h5f = h5py.File(str(dataset_loc + '/'+dataset_name),'w')
h5f.create_dataset('dataset', data=train_dataset)
h5f.close()


print('\n--------------------------------------------------------------')
print('Saving validation dataset:', validation_dataset_name)
print('--------------------------------------------------------------')

h5f = h5py.File(str(dataset_loc + '/'+validation_dataset_name),'w')
h5f.create_dataset('dataset', data=validation_dataset)
h5f.close()
