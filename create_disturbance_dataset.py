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
import difflib
import datetime
import random



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


def find_nearest_timestamp(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def timeSinceEpoch(filename):
    fn_path,fn = os.path.split(filename)

    fn_date = (fn_path.split('/')[2]).split('-')

    fn = os.path.splitext(fn)[0]
    h,m,s = fn.split('_')

    # print(int(datetime.timedelta(hours=int(dstrb_h),minutes=int(dstrb_m),seconds=int(dstrb_s)).total_seconds()))
    timestamp = int(datetime.datetime(int(fn_date[0]), int(fn_date[1]),int(fn_date[2]),int(h),int(m),int(s)).timestamp())
    return timestamp


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


def normalise_dataset(dataset):
    mean = np.mean(dataset,axis=1)
    std = np.std(dataset,axis=1)

    dataset = (dataset-mean[:,np.newaxis])/std[:,np.newaxis]
    return [dataset,mean,std]

def minmax_feature_scaling(dataset):
    max = np.amax(dataset,axis=1)
    min = np.amin(dataset,axis=1)
    dataset = dataset-min[:,np.newaxis]/((max-min)[:,np.newaxis])
    return dataset,max,min


def generateDataset(list_ulogs,list_disturbance_logs):

    features = 33
    dataset_num_entries=10000000
    counter = 0
    dataset = np.zeros((features,dataset_num_entries))
    ulog_entry_prev = ''


    log_progressbar = trange(len(list_ulogs), desc='', leave=True)

    # for ulog_entry in tqdm(logs, desc = "log file: {}".format(str(ulog_entry_prev))):
    for ulog_timestamp in list_ulogs:
        disturbance_log = ''
        ulog_entry = ''
        dstrb_timestamps = list(disturbance_logs.keys())
        nearest_timestamp = find_nearest_timestamp( dstrb_timestamps,ulog_timestamp )

        if(np.abs(nearest_timestamp - ulog_timestamp) < 10):
            disturbance_log = disturbance_logs.get(nearest_timestamp)
            ulog_entry = list_ulogs.get(ulog_timestamp)

        else:
            print("no match found for Ulog: {}".format(list_ulogs.get(ulog_timestamp)))

        # print(ulog_entry)

        log_progressbar.set_description("log: {} {}".format(ulog_entry,disturbance_log))
        log_progressbar.refresh() # to show immediately the update
        log_progressbar.update()

        listOfCSVs = convert_ulog2csv(ulog_entry,log_eoi,'./',',')

        # print(listOfCSVs)





        # disturbance
        df = pd.read_csv(disturbance_log)
        # time 10^6, since px4 does logging in microseconds
        timestamp_disturb = np.floor(np.array(df['timestamp'].tolist())*1000000).astype(int)
        fx = np.array(df['fx'].tolist())
        fy = np.array(df['fy'].tolist())
        fz = np.array(df['fz'].tolist())
        mx = np.array(df['mx'].tolist())
        my = np.array(df['my'].tolist())
        mz = np.array(df['mz'].tolist())


        # vehicle_attitude csv

        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[1]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)

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
        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[0]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)
        timestamp_local_position = np.array(df['timestamp'].tolist())
        vx = np.array(df['vx'].tolist()).astype(float)
        vy = np.array(df['vy'].tolist()).astype(float)
        vz = np.array(df['vz'].tolist()).astype(float)
        ax = np.array(df['ax'].tolist()).astype(float)
        ay = np.array(df['ay'].tolist()).astype(float)
        az = np.array(df['az'].tolist()).astype(float)
        size_local_position = timestamp_local_position.size


        # vechile_actuator_outputs csv
        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[2]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)
        timestamp_actuator_outputs = np.array(df['timestamp'].tolist())
        timestamp_actuator_outputs_cor = np.zeros(length_attitude)
        PWM_0 = np.array(df['output[0]'].tolist()).astype(float)
        PWM_1 = np.array(df['output[1]'].tolist()).astype(float)
        PWM_2 = np.array(df['output[2]'].tolist()).astype(float)
        PWM_3 = np.array(df['output[3]'].tolist()).astype(float)

# log_eoi = 'vehicle_local_position,vehicle_attitude,actuator_outputs,\
#             vehicle_local_position_setpoint,vehicle_rates_setpoint,\
#             vehicle_attitude_setpoint'


        # vechile_local_position_setpoint csv
        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[3]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)
        timestamp_local_position_setpoint = np.array(df['timestamp'].tolist())
        ref_vx = np.array(df['vx'].tolist()).astype(float)
        ref_vy = np.array(df['vy'].tolist()).astype(float)
        ref_vz = np.array(df['vz'].tolist()).astype(float)
        size_local_position_setpoint = timestamp_local_position_setpoint.size


        # vechile_rates_setpoint csv
        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[4]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)
        timestamp_vehicle_rate_setpoint = np.array(df['timestamp'].tolist())
        ref_P = np.array(df['roll'].tolist()).astype(float)
        ref_Q = np.array(df['pitch'].tolist()).astype(float)
        ref_R = np.array(df['yaw'].tolist()).astype(float)
        size_vehicle_rate_setpoint = timestamp_vehicle_rate_setpoint.size


        # vehicle_attitude_setpoint csv
        csv_name = str(difflib.get_close_matches(str("_"+listOfEOI[5]+"_0"), listOfCSVs)[0])
        df = pd.read_csv(csv_name)
        timestamp_vehicle_attitude_setpoint = np.array(df['timestamp'].tolist())
        q1_d = np.array(df['q_d[0]'].tolist()).astype(float)
        q2_d = np.array(df['q_d[1]'].tolist()).astype(float)
        q3_d = np.array(df['q_d[2]'].tolist()).astype(float)
        q4_d = np.array(df['q_d[3]'].tolist()).astype(float)
        q_d = [q1_d,q2_d,q3_d,q4_d]
        size_vehicle_attitude_setpoint = timestamp_vehicle_attitude_setpoint.size


        # not all entries are written at the same time in the log file, thus we
        # will align them all by assign them to closet entry that corresponds in the
        # local_position log

        # timestamp_local_position = timestamp_attitude
        # size_local_position = length_attitude

        P_ = np.zeros(len(timestamp_local_position),dtype=float)
        Q_ = np.zeros(len(timestamp_local_position),dtype=float)
        R_ = np.zeros(len(timestamp_local_position),dtype=float)
        q1_ = np.zeros(len(timestamp_local_position),dtype=float)
        q2_ = np.zeros(len(timestamp_local_position),dtype=float)
        q3_ = np.zeros(len(timestamp_local_position),dtype=float)
        q4_ = np.zeros(len(timestamp_local_position),dtype=float)
        q1_d_ = np.zeros(len(timestamp_local_position),dtype=float)
        q2_d_ = np.zeros(len(timestamp_local_position),dtype=float)
        q3_d_ = np.zeros(len(timestamp_local_position),dtype=float)
        q4_d_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_vx_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_vy_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_vz_ = np.zeros(len(timestamp_local_position),dtype=float)
        vx_ = np.zeros(len(timestamp_local_position),dtype=float)
        vy_ = np.zeros(len(timestamp_local_position),dtype=float)
        vz_ = np.zeros(len(timestamp_local_position),dtype=float)
        ax_ = np.zeros(len(timestamp_local_position),dtype=float)
        ay_ = np.zeros(len(timestamp_local_position),dtype=float)
        az_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_P_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_Q_ = np.zeros(len(timestamp_local_position),dtype=float)
        ref_R_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_0_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_1_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_2_ = np.zeros(len(timestamp_local_position),dtype=float)
        PWM_3_ = np.zeros(len(timestamp_local_position),dtype=float)
        fx_ = np.zeros(len(timestamp_local_position),dtype=float)
        fy_ = np.zeros(len(timestamp_local_position),dtype=float)
        fz_ = np.zeros(len(timestamp_local_position),dtype=float)
        mx_ = np.zeros(len(timestamp_local_position),dtype=float)
        my_ = np.zeros(len(timestamp_local_position),dtype=float)
        mz_ = np.zeros(len(timestamp_local_position),dtype=float)

        time_keeper = 0
        for value in timestamp_local_position:

            if(value >= 20000000 and value <= 50000000):
                loc = find_nearest(timestamp_disturb,value)

                fx_[time_keeper] = fx[loc]
                fy_[time_keeper] = fy[loc]
                fz_[time_keeper] = fz[loc]
                mx_[time_keeper] = mx[loc]
                my_[time_keeper] = my[loc]
                mz_[time_keeper] = mz[loc]


            loc = find_nearest(timestamp_actuator_outputs,value)
            PWM_0_[time_keeper] = PWM_0[loc]
            PWM_1_[time_keeper] = PWM_1[loc]
            PWM_2_[time_keeper] = PWM_2[loc]
            PWM_3_[time_keeper] = PWM_3[loc]

            loc = find_nearest(timestamp_attitude,value)
            P_[time_keeper] = P[loc]
            Q_[time_keeper] = Q[loc]
            R_[time_keeper] = R[loc]
            q1_[time_keeper] = q1[loc]
            q2_[time_keeper] = q2[loc]
            q3_[time_keeper] = q3[loc]
            q4_[time_keeper] = q4[loc]

            loc = find_nearest(timestamp_vehicle_attitude_setpoint,value)
            q1_d_[time_keeper] = q1_d[loc]
            q2_d_[time_keeper] = q2_d[loc]
            q3_d_[time_keeper] = q3_d[loc]
            q4_d_[time_keeper] = q4_d[loc]

            loc = find_nearest(timestamp_vehicle_rate_setpoint,value)
            ref_P_[time_keeper] = ref_P[loc]
            ref_Q_[time_keeper] = ref_Q[loc]
            ref_R_[time_keeper] = ref_R[loc]

            loc = find_nearest(timestamp_local_position_setpoint,value)
            ref_vx_[time_keeper] = ref_vx[loc]
            ref_vy_[time_keeper] = ref_vy[loc]
            ref_vz_[time_keeper] = ref_vz[loc]


            loc = find_nearest(timestamp_local_position,value)
            vx_[time_keeper] = vx[loc]
            vy_[time_keeper] = vy[loc]
            vz_[time_keeper] = vz[loc]
            ax_[time_keeper] = ax[loc]
            ay_[time_keeper] = ay[loc]
            az_[time_keeper] = az[loc]

            time_keeper += 1


        q_ = [q1_,q2_,q3_,q4_]



        T1 = (PWM_0_/1000 - 1)*motor_thrust*9.81
        T2 = (PWM_1_/1000 - 1)*motor_thrust*9.81
        T3 = (PWM_2_/1000 - 1)*motor_thrust*9.81
        T4 = (PWM_3_/1000 - 1)*motor_thrust*9.81


        dataset[0,counter:counter  + size_local_position] = q1_
        dataset[1,counter:counter  + size_local_position] = q2_ # ~ roll angle
        dataset[2,counter:counter  + size_local_position] = q3_ # ~ pitch angle
        dataset[3,counter:counter  + size_local_position] = q4_
        dataset[4,counter:counter  + size_local_position] = q1_d_
        dataset[5,counter:counter  + size_local_position] = q2_d_ # ~ roll angle
        dataset[6,counter:counter  + size_local_position] = q3_d_ # ~ pitch angle
        dataset[7,counter:counter  + size_local_position] = q4_d_
        dataset[8,counter:counter  + size_local_position] = vx_
        dataset[9,counter:counter  + size_local_position] = vy_
        dataset[10,counter:counter  + size_local_position] = vz_
        dataset[11,counter:counter  + size_local_position] = ref_vx_
        dataset[12,counter:counter  + size_local_position] = ref_vy_
        dataset[13,counter:counter  + size_local_position] = ref_vz_
        dataset[14,counter:counter  + size_local_position] = T1
        dataset[15,counter:counter  + size_local_position] = T2
        dataset[16,counter:counter  + size_local_position] = T3
        dataset[17,counter:counter + size_local_position] = T4
        dataset[18,counter:counter + size_local_position] = P_
        dataset[19,counter:counter + size_local_position] = Q_
        dataset[20,counter:counter + size_local_position] = R_
        dataset[21,counter:counter + size_local_position] = ref_P_
        dataset[22,counter:counter + size_local_position] = ref_Q_
        dataset[23,counter:counter + size_local_position] = ref_R_
        dataset[24,counter:counter + size_local_position] = ax_
        dataset[25,counter:counter + size_local_position] = ay_
        dataset[26,counter:counter + size_local_position] = az_
        dataset[27,counter:counter + size_local_position] = fx_
        dataset[28,counter:counter + size_local_position] = fy_
        dataset[29,counter:counter + size_local_position] = fz_
        dataset[30,counter:counter + size_local_position] = mx_
        dataset[31,counter:counter + size_local_position] = my_
        dataset[32,counter:counter + size_local_position] = mz_


        for csvFile in listOfCSVs:
            os.remove(csvFile)

        # perhaps delete *.csv
        counter += size_local_position

        if(counter + 3*size_local_position >= dataset_num_entries):
            dataset_num_entries += 1000000
            dataset = np.hstack((dataset,np.zeros((features,1000000))))

    # resize dataset by using counter variable
    dataset = np.delete(dataset,slice(counter,dataset_num_entries,1),1)
    return dataset

# log directory
# log_dir = "/home/henry/esl-sun/PX4/build/px4_sitl_default/logs/"
log_dir = "./logs/"
# entries of interest
log_eoi = 'vehicle_local_position,vehicle_attitude,actuator_outputs,vehicle_local_position_setpoint,vehicle_rates_setpoint,vehicle_attitude_setpoint'
listOfEOI = log_eoi.split(',')


description=''




numOfLogs=0
numOfValLogs=0
listOfULogs=[]
listOfDisturbances=[]


# find all *.ulog files in log directory
for root, dirs, files in os.walk(log_dir):
    for name in files:
        log_name = str(os.path.join(root,name))
        # extension
        ext = str(os.path.splitext(name)[1])

        if(ext == '.ulg'):
            listOfULogs.append(log_name)
            numOfLogs = numOfLogs + 1

        elif(ext == '.dist'):
            listOfDisturbances.append(log_name)

# convert all found *.ulog files into *.cvs type format and store in desired dir
# with messages of interest
# for ulog_entru in tqdm(listOfLogs):
#     print(ulog_entru)

numOfValLogs = int(np.floor(numOfLogs*validation_percentage))

training_logs = {}
validation_logs = {}
disturbance_logs = {}


for dstrb_log in listOfDisturbances:
    timestamp = timeSinceEpoch(dstrb_log)
    disturbance_logs.update({timestamp:dstrb_log})

listOfValLogs = random.choices(listOfULogs,k=numOfValLogs)

for validation_log in listOfValLogs:
    timestamp = timeSinceEpoch(validation_log)
    validation_logs.update({timestamp:validation_log})


for log in listOfULogs:
    if log not in list(validation_logs.values()):
        timestamp = timeSinceEpoch(log)
        training_logs.update({timestamp:log})


train_dataset = generateDataset(training_logs,listOfDisturbances)
validation_dataset = generateDataset(validation_logs,listOfDisturbances)

train_dataset = np.nan_to_num(train_dataset)
validation_dataset = np.nan_to_num(validation_dataset)

train_num_samples=len(train_dataset[0,:])
val_num_samples=len(validation_dataset[0,:])


# train_dataset,train_dataset_max,train_dataset_min = minmax_feature_scaling(train_dataset)
# validation_dataset,validation_dataset_max,validation_dataset_min = minmax_feature_scaling(validation_dataset)


train_dataset,mean,std = normalise_dataset(train_dataset)
validation_dataset,val_mean,val_std = normalise_dataset(validation_dataset)


train_dataset,train_min,train_max = minmax_feature_scaling(train_dataset)
max_validation_dataset_values,val_min,val_max = minmax_feature_scaling(validation_dataset)

train_dataset = np.nan_to_num(train_dataset)
validation_dataset = np.nan_to_num(validation_dataset)

# train_dataset = train_dataset/max_train_dataset_values[:,np.newaxis]
# train_dataset,meep = whiten(train_dataset)
# [x,y,z] = whitening_dataset(train_dataset)
# print(y,z)

print('\n--------------------------------------------------------------')
print('Saving dataset readme at:', str(dataset_loc + '/'+dataset_name+'_readme'))
print('--------------------------------------------------------------')
with shelve.open( str(dataset_loc + '/'+dataset_name+'_readme')) as db:


    db['std'] = std
    db['mean'] = mean
    # db['max_train_dataset_values'] = train_dataset_max
    # db['max_validation_dataset_values'] = validation_dataset_max
    # db['min_train_dataset_values'] = train_dataset_min
    # db['min_validation_dataset_values'] = validation_dataset_min

    db['train_dataset_num_entries'] = train_num_samples
    db['validation_dataset_num_entries'] = val_num_samples
    db['numOfLogs'] = len(listOfULogs)
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
