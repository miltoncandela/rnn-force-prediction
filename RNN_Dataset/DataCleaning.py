# Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime

# Input data
id = 'S4R3'
subject = 'S4'
repetition = 'R3'
video_timestamp = '2021/10/15/19:19:42' 
video_frameDuration = 0.03340440613 # seconds

#-------------------------------------------

realsense_keypoints = pd.read_csv("skeletontracking/"+id+"-KeyPoints.csv")
empatica_acc = pd.read_csv("empatica_acc/"+id+"_ACC.csv")

#------- SKELETON TRACKING DATA -------

# Create timestamp column for keypoints data
timestamp_keypoints_diferences = datetime.timedelta(seconds = video_frameDuration)
timestamp_keypoints = pd.Series([pd.to_datetime(video_timestamp)])
for i in range(1,realsense_keypoints.shape[0]):
    timestamp_keypoints[i]= (timestamp_keypoints[i-1] + timestamp_keypoints_diferences).round(freq='30L')

# Create keypoints dataframe
keypoints_df = pd.DataFrame()
ImportantKeypoints = ["Nose","Neck","Right_Shoulder","Right_Elbow","Right_Wrist","Left_Shoulder","Left_Elbow","Left_Wrist","Right_Hip","Right_Knee","Right_Ankle","Left_Hip","Left_Knee","Left_Ankle"]
for keypoint in ImportantKeypoints:
    Coordinates = realsense_keypoints.loc[:,keypoint]
    X = Coordinates.apply(lambda x: re.search('(?<=x=)(.*)(?=,)',x).group(0)).astype(float)
    Y = Coordinates.apply(lambda x: re.search('(?<=y=)(.*)(?=\))',x).group(0)).astype(float)
    keypoints_df[keypoint+"_X"] = X
    keypoints_df[keypoint+"_Y"] = Y

# replace invalid keypoints
keypoints_df.replace(-1, np.NaN, inplace=True)
keypoints_df.interpolate(axis=0, inplace=True)

# Add subject, repetition and timestamp columns
keypoints_df.insert(0,'Subject', [subject for i in range(realsense_keypoints.shape[0])])
keypoints_df.insert(1,'Repetition', [repetition for i in range(realsense_keypoints.shape[0])])
keypoints_df.insert(2, 'Datetime', timestamp_keypoints)

keypoints_df.to_csv("Final DataFrames/"+id+'_final_keypoints')

#------- EMPATICA DATA -------

# create dataframe with aceleration data
empatica_acc_df = pd.DataFrame()
empatica_acc_df['ACC_x'] = empatica_acc['valueACC'].apply(lambda x: re.search('(?<=\[)([-\d]*)(?=, )',x).group(0)).astype(float)
empatica_acc_df['ACC_y'] = empatica_acc['valueACC'].apply(lambda x: re.search('(?<=, )([-\d]*)(?=, )',x).group(0)).astype(float)
empatica_acc_df['ACC_z'] = empatica_acc['valueACC'].apply(lambda x: re.search('(?<=, )([-\d]*)(?=\])',x).group(0)).astype(float)
empatica_acc_df['Datetime'] = pd.to_datetime(empatica_acc['Datetime']).apply(lambda x: x.round(freq='30L'))

empatica_acc_df.to_csv("Final DataFrames/"+id+'_final_acc')

#------- MERGE DATA -------

df1 = keypoints_df.merge(empatica_acc_df, how='outer', on = 'Datetime', sort = True)

start = empatica_acc_df['Datetime'][0] if keypoints_df['Datetime'][0] <= empatica_acc_df['Datetime'][0] else keypoints_df['Datetime'][0]
end = empatica_acc_df['Datetime'].iloc[-1] if keypoints_df['Datetime'].iloc[-1] >= empatica_acc_df['Datetime'].iloc[-1] else keypoints_df['Datetime'].iloc[-1]

final_df = df1.set_index('Datetime').loc[start:end]
final_df.interpolate(axis=0, inplace=True)

final_df.to_csv("Final_DataFrames/"+id+'_final_df')

#------- PLOT DATA -------
plt.figure()
plt.plot(final_df.loc[:,'Nose_X':'ACC_z'])
plt.xlabel('time')
plt.legend(final_df.columns[2:], fontsize='xx-small', loc='right')
plt.title("Final_DataFrames/"+id + ' - Final Data Frame')
plt.xlim(start,end+(end-start)/4)
plt.show()
