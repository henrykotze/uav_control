import os
import pandas as pd
import numpy as np
import difflib


log_dir = "/home/henry/esl-sun/PX4/build/px4_sitl_default/logs"
#
#
# listOfLogs = []

listOfUlogs = []
listOfDisturbances = []

for root, dirs, files in os.walk(log_dir):
    for name in files:

        log_name = str(os.path.join(root,name))
        # get extension
        ext = str(os.path.splitext(name)[1])
        # print(ext)
        # print(files)
        if(ext == '.ulg'):
            listOfUlogs.append(log_name)

        elif(ext == '.dist'):
            listOfDisturbances.append(log_name)



print(listOfDisturbances)
print(listOfUlogs)

for ulog_entry in listOfUlogs:
    disturbance_log = difflib.get_close_matches(ulog_entry, listOfDisturbances,n=1)

    # ext = str(os.path.splitext(disturbance_log))
    # print(ext)


    print(ulog_entry,disturbance_log)




df = pd.read_csv(listOfDisturbances[0])

timestamp_disturb = np.array(df['timestamp'].tolist())
fx = np.array(df['fx'].tolist())
fy = np.array(df['fy'].tolist())
fz = np.array(df['fz'].tolist())
mx = np.array(df['mx'].tolist())
my = np.array(df['my'].tolist())
mz = np.array(df['mz'].tolist())

print(fx)
