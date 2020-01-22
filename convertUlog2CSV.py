import numpy as np
import ulog2csv
import os
from core import ULog
import pandas as pd
import math
from tqdm import trange, tqdm
import argparse



parser = argparse.ArgumentParser(\
        prog='Generates h5py dataset from PX4 Ulog files',\
        description='Creates .npz files containing the responses of a rotary wing UAV'
        )


parser.add_argument('-logdir', default='./logs/', help='location to store responses')
parser.add_argument('-drone', default='./griffin.txt', help='path to file containing drone info')
parser.add_argument('-add_info', default='', help='path to file containing drone info')
parser.add_argument('-dataset_name', default='', help='path to file containing drone info')

args = parser.parse_args()



logdir = vars(args)['logdir']
drone_file_path = str(vars(args)['drone'])
addition_info = str(vars(args)['add_info'])
dataset_name = str(vars(args)['dataset_name'])

drone_file = open(drone_file_path,"r")
drone_str = drone_file.readlines()[2].split(";")[0:-1]
drone_info = [str(i) for i in drone_str]
drone_file.close()


print("===============================")
print("DRONE INFO")
print("===============================")
print("{:<15} {:<10}".format("name:", drone_info[0]))
print("{:<15} {:<10}".format("mass:", drone_info[1]))
print("{:<15} {:<10}".format("Ixx:", drone_info[2]))
print("{:<15} {:<10}".format("Iyy:", drone_info[3]))
print("{:<15} {:<10}".format("Izz:", drone_info[4]))
print("{:<15} {:<10}".format("R_Ld:", drone_info[5]))
print("{:<15} {:<10}".format("r_D:", drone_info[6]))
print("{:<15} {:<10}".format("tau:", drone_info[7]))
print("{:<15} {:<10}".format("motor thrust:", drone_info[8]))
print("{:<15} {:<10}".format("config:", drone_info[9]))
print("===============================")
print("===============================")

motor_thrust = float(drone_info[8])

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

# log directory
log_dir = "/home/henry/esl-sun/PX4/build/px4_sitl_default/logs/"
# entries of interest
log_eoi = 'vehicle_local_position,vehicle_attitude,actuator_outputs'
listOfEOI = log_eoi.split(',')


dataset_locations = './datasets'
description=''
dataset_name=''

features = 17
dataset_num_entries=10000000
counter = 0
dataset = np.zeros((features,dataset_num_entries),float)



numOfLogs=0
listOfLogs=[]


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

for ulog_entry in tqdm(listOfLogs):


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
    P_ = np.zeros(len(timestamp_local_position))
    Q_ = np.zeros(len(timestamp_local_position))
    R_ = np.zeros(len(timestamp_local_position))
    q1_ = np.zeros(len(timestamp_local_position))
    q2_ = np.zeros(len(timestamp_local_position))
    q3_ = np.zeros(len(timestamp_local_position))
    q4_ = np.zeros(len(timestamp_local_position))
    PWM_0_ = np.zeros(len(timestamp_local_position))
    PWM_1_ = np.zeros(len(timestamp_local_position))
    PWM_2_ = np.zeros(len(timestamp_local_position))
    PWM_3_ = np.zeros(len(timestamp_local_position))

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


    dataset[0,counter:counter+size_local_position] = P_
    dataset[1,counter:counter+size_local_position] = Q_
    dataset[2,counter:counter+size_local_position] = R_
    dataset[3,counter:counter+size_local_position] = q1_
    dataset[4,counter:counter+size_local_position] = q2_
    dataset[5,counter:counter+size_local_position] = q3_
    dataset[6,counter:counter+size_local_position] = q4_
    dataset[7,counter:counter+size_local_position] = PWM_0_
    dataset[8,counter:counter+size_local_position] = PWM_1_
    dataset[9,counter:counter+size_local_position] = PWM_2_
    dataset[10,counter:counter + size_local_position] = PWM_3_
    dataset[11,counter:counter + size_local_position] = U_
    dataset[12,counter:counter + size_local_position] = V_
    dataset[13,counter:counter + size_local_position] = W_
    dataset[14,counter:counter + size_local_position] = Udot
    dataset[15,counter:counter + size_local_position] = Vdot
    dataset[16,counter:counter + size_local_position] = Wdot


    # print(counter, size_local_position)


    for csvFile in listOfCSVs:
        os.remove(csvFile)

    # perhaps delete *.csv
    counter += size_local_position


# resize dataset by using counter variable


# with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme')) as db:
#     db['maxPdot'] = maxPdot
#     db['maxQdot'] = maxQdot
#     db['maxRdot'] = maxRdot
#
#     db['maxUdot'] = maxUdot
#     db['maxVdot'] = maxVdot
#     db['maxWdot'] = maxWdot
#
#     db['maxP'] = maxP
#     db['maxQ'] = maxQ
#     db['maxR'] = maxR
#
#     db['maxU'] = maxU
#     db['maxV'] = maxV
#     db['maxW'] = maxW
#
# db.close()


print('\n--------------------------------------------------------------')
print('Saving features and labels to:', nameOfDataset)
print('--------------------------------------------------------------')

h5f = h5py.File(str(dataset_loc + '/'+nameOfDataset),'w')
h5f.create_dataset('dataset', data=dataset)
h5f.close()
