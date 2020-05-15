import os
import pandas as pd
import numpy as np


# log_dir = "./logs/"
#
#
# listOfLogs = []
# listOfUlogs = []
# listOfDisturbances = []
#
# for root, dirs, files in os.walk(log_dir):
#     for name in files:
#
#         log_name = str(os.path.join(root,name))
#         # get extension
#         ext = str(os.path.splitext(name)[1])
#         # print(ext)
#         # print(files)
#         if(ext == '.ulg'):
#             listOfUlogs.append(log_name)
#
#         elif(ext == '.csv'):
#             listOfDisturbances.append(log_name)
#
#
#
# print(listOfDisturbances)
# print(listOfUlogs)

disturbance_log = "./test.csv"
df = pd.read_csv(disturbance_log)


timestamp_disturb = np.array(df['time'].tolist())
fx = np.array(df['fx'].tolist())
fy = np.array(df['fy'].tolist())
fz = np.array(df['fz'].tolist())
mx = np.array(df['mx'].tolist())
my = np.array(df['my'].tolist())
mz = np.array(df['mz'].tolist())


print(fx)
