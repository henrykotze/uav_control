


import os
import difflib
import numpy as np
import datetime
import math
import random



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


def check_logs(list_ulogs,disturbance_logs):

    # log_progressbar = range(len(list_ulogs), desc='', leave=True)

    # for ulog_entry in tqdm(logs, desc = "log file: {}".format(str(ulog_entry_prev))):
    for ulog_timestamp in list_ulogs:


        meep = list(disturbance_logs.keys())
        # print(meep)
        nearest_timestamp = find_nearest_timestamp( meep,ulog_timestamp )

        if(np.abs(nearest_timestamp - ulog_timestamp) < 10):
            print(disturbance_logs.get(nearest_timestamp),list_ulogs.get(ulog_timestamp))

        else:
            print("WARNING")


log_dir = "./logs/"
# entries of interest
log_eoi = 'vehicle_local_position,vehicle_attitude,actuator_outputs,vehicle_local_position_setpoint,vehicle_rates_setpoint,vehicle_attitude_setpoint'
listOfEOI = log_eoi.split(',')

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
validation_percentage=0.1
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


# print(training_logs.keys())
# print('===============================')
# print(validation_logs.keys())
#
# print('===============================')
# print(list(disturbance_logs.keys()))
# check_logs(training_logs,disturbance_logs)

check_logs(training_logs,disturbance_logs)
