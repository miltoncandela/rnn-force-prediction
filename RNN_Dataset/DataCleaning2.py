# Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime
import cv2
from scipy.signal import butter,filtfilt

# -------------------------------------------------------------------------------------
#   INPUT DATA: 
# -------------------------------------------------------------------------------------

id = 'S3_M1_R2'
subject = 3
movement = 1
repetition = 2
video_timestamp = '2022/04/26/16:25:07'
videoname = 'VID_20220426_162507'

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# ------- VIDEO PROPERTIES ------------------------------------------------------------
cap = cv2.VideoCapture('./DATA/videos/'+videoname+'.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_frameDuration = 1/fps # seconds
video_framewidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
print(video_framewidth)

# ------- READ FILES ----------------------
realsense_keypoints = pd.read_csv("./DATA/skeleton/"+id+"-KeyPoints.csv")
acc_brazo_df = pd.read_csv("./DATA/acc_brazo/"+id+"_B-acc.csv")
acc_pierna_df = pd.read_csv("./DATA/acc_pierna/"+id+"_P-acc.csv")

# ------- MATLAB ACCELERATION DATA ----------------------
#Brazo
acc_brazo_df['T'] = pd.to_datetime(acc_brazo_df['T'])
acc_brazo_df.columns = ['Datetime','Brazo_acc_x','Brazo_acc_y','Brazo_acc_z']
acc_brazo_df.set_index('Datetime', inplace = True)
acc_brazo_df_upsamp = acc_brazo_df.resample("10L").mean()
acc_brazo_df_upsamp.interpolate(method='spline', order=2, axis=0, inplace=True)

#Pierna
acc_pierna_df['T'] = pd.to_datetime(acc_pierna_df['T'])
acc_pierna_df.columns = ['Datetime','Pierna_acc_z','Pierna_acc_y','Pierna_acc_x']
acc_pierna_df.set_index('Datetime', inplace = True)
acc_pierna_df_upsamp = acc_pierna_df.resample("10L").mean()
acc_pierna_df_upsamp.interpolate(method='spline', order=2, axis=0, inplace=True)

# Filter Data
normal_cutoff = 15/(0.5*100)  # 100=sambple frecuency, 8=cutoff frecuency
b, a = butter(8, normal_cutoff, btype='low', analog=False)  # Get the filter coefficients

brazo_df_filtered = pd.DataFrame()
for column in acc_brazo_df_upsamp.columns:
    brazo_df_filtered[column] = filtfilt(b,a,acc_brazo_df_upsamp[column])
brazo_df_filtered.columns = acc_brazo_df_upsamp.columns
brazo_df_filtered.index = acc_brazo_df_upsamp.index

pierna_df_filtered = pd.DataFrame()
for column in acc_pierna_df_upsamp.columns:
    pierna_df_filtered[column] = filtfilt(b,a,acc_pierna_df_upsamp[column])
pierna_df_filtered.columns = acc_pierna_df_upsamp.columns
pierna_df_filtered.index = acc_pierna_df_upsamp.index

def plot_acceleration(dfname, colname1, colnname2, colname3, title):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, constrained_layout=True)

    ax1.plot(dfname[colname1], c='#0072BD')
    ax1.set_title(id + ' - ' + title)
    ax1.set_ylabel ('m/s^2')
    ax1.legend(['acc X'])

    ax2.plot(dfname[colnname2], c='#D95319')
    ax2.set_ylabel ('m/s^2')
    ax2.legend(['acc Y'])

    ax3.plot(dfname[colname3], c='#7E2F8E')
    ax3.set_xlabel('Time')
    ax3.set_ylabel ('m/s^2')
    ax3.legend(['acc Z'])

    plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - ' +title)
    plt.show()

plot_acceleration(brazo_df_filtered, 'Brazo_acc_x', 'Brazo_acc_y', 'Brazo_acc_z', 'Arm Acceleration')
plot_acceleration(pierna_df_filtered, 'Pierna_acc_x', 'Pierna_acc_y', 'Pierna_acc_z', 'Leg Acceleration')

#------- SKELETON TRACKING DATA ---------

# Create timestamp column for keypoints data
timestamp_keypoints_diferences = datetime.timedelta(seconds = video_frameDuration)
timestamp_keypoints = pd.Series([pd.to_datetime(video_timestamp)])
for i in range(1,realsense_keypoints.shape[0]):
    timestamp_keypoints[i]= (timestamp_keypoints[i-1] + timestamp_keypoints_diferences)
    
# Create keypoints dataframe
keypoints_df = pd.DataFrame()
ImportantKeypoints = ["Nose","Neck","Right_Shoulder","Right_Elbow","Right_Wrist","Left_Shoulder","Left_Elbow","Left_Wrist","Right_Hip","Right_Knee","Right_Ankle","Left_Hip","Left_Knee","Left_Ankle"]
for keypoint in ImportantKeypoints:
    Coordinates = realsense_keypoints.loc[:,keypoint]
    X = Coordinates.apply(lambda x: re.search('(?<=x=)(.*)(?=,)',x).group(0)).astype(float)
    Y = Coordinates.apply(lambda x: re.search('(?<=y=)(.*)(?=\))',x).group(0)).astype(float)
    keypoints_df[keypoint+"_X"] = X
    keypoints_df[keypoint+"_Y"] = Y
keypoints_df.insert(0, 'Datetime', timestamp_keypoints)

# replace or delate invalid keypoints
keypoints_df.set_index(keypoints_df.Nose_X==-2,inplace=True)
keypoints_df = keypoints_df.loc[False]
#keypoints_df.drop(True, axis=0,inplace=True)
keypoints_df.set_index('Datetime',drop=True, inplace = True)
keypoints_df.replace(-1, np.NaN, inplace=True)

for i in keypoints_df.index:
    if (keypoints_df.loc[i].isnull().sum()>0):
        keypoints_df.drop(i,axis=0,inplace=True)
    else:
        break
for i in keypoints_df.index[::-1]:
    if (keypoints_df.loc[i].isnull().sum()>0):
        keypoints_df.drop(i,axis=0,inplace=True)
    else:
        break

keypoints_df.interpolate(method='spline', order=2, axis=0, inplace=True)
keypoints_df.dropna(axis=0, inplace=True)

# Upsampling
keypoints_df_upsamp = keypoints_df.resample("10L").mean()
keypoints_df_upsamp.interpolate(method='linear', order=2, axis=0, inplace=True)

# Filter Data
normal_cutoff = 5/(0.5*100)  # 100=sambple frecuency, 8=cutoff frecuency
b, a = butter(8, normal_cutoff, btype='low', analog=False)  # Get the filter coefficients

keypoints_df_filtered = pd.DataFrame()
for keypoint in keypoints_df_upsamp.columns:
    keypoints_df_filtered[keypoint] = filtfilt(b,a,keypoints_df_upsamp[keypoint])
keypoints_df_filtered.columns = keypoints_df_upsamp.columns
keypoints_df_filtered.index = keypoints_df_upsamp.index

# Convert to meters
keypoints_df_escaled = keypoints_df_filtered/video_framewidth*6

# Remove offset
y_labels = keypoints_df_escaled.columns[np.arange(1,28,2)]
x_labels = keypoints_df_escaled.columns[np.arange(0,28,2)]
keypoints_df_escaled[y_labels] = keypoints_df_escaled[y_labels]*-1
offset = keypoints_df_escaled['Right_Ankle_Y'].min()
keypoints_df_escaled[y_labels] = keypoints_df_escaled[y_labels] - offset

# Graphs processing of the signal

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, constrained_layout=True)

ax1.plot(keypoints_df['Right_Knee_Y'], c='#0072BD')
ax1.set_title(id + ' - ' + 'Right_Knee_Y')
ax1.set_ylabel ('m')
ax1.legend(['Original'])

ax2.plot(keypoints_df_upsamp['Right_Knee_Y'], c='#D95319')
ax2.set_ylabel ('m')
ax2.legend(['Upsampled'])

ax3.plot(keypoints_df_filtered['Right_Knee_Y'], c='#7E2F8E')
ax3.set_xlabel('Time')
ax3.set_ylabel ('m')
ax3.legend(['Filtered'])

plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - Preprocessing')
plt.show()

# Figures of skeleton data

plt.figure()
plt.plot(keypoints_df_escaled[y_labels])
plt.xlabel('Time')
plt.ylabel('Position (m)')
start = keypoints_df_escaled.index[0]
end = keypoints_df_escaled.index[-1]
plt.xlim(right=(end+(end-start)/2))
plt.legend(y_labels, fontsize='small', loc='right')
plt.title(id + ' - Keypoints eje Y')
plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - Keypoints Y')
plt.show()

plt.figure()
plt.plot(keypoints_df_escaled[x_labels])
plt.xlabel('Time')
plt.ylabel('Position (m)')
plt.xlim(right=(end+(end-start)/2))
plt.legend(x_labels, fontsize='small', loc='right')
plt.title(id + ' - Keypoints eje X')
plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - Keypoints X')
plt.show()

# Join all data in a single DataFrame
final_df = keypoints_df_escaled.join(pierna_df_filtered, how='inner').join(brazo_df_filtered, how='inner')
if movement == 1:
    final_df = final_df.iloc[0:700]
elif movement == 2:
    final_df = final_df.iloc[0:400]
elif movement == 3:
    final_df = final_df.iloc[10:660]

plt.figure()
plt.plot(final_df)
plt.xlabel('time')
plt.legend(final_df.columns, fontsize='xx-small', loc='right')
plt.title(id + ' - Final Data Frame')
start = final_df.index[0]
end = final_df.index[-1]
plt.xlim(right=(end+(end-start)/3))
plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - Final Data Frame')
plt.show()

def plot_final(dfname, colname1, colname2, colname3, colname4, title):
    f, ax = plt.subplots(2, 2, sharex=True, constrained_layout=True)

    ax[0,0].plot(dfname[colname1], c='#0072BD')
    ax[0,0].set_title(id + ' - ' + title + ' - X ')
    ax[0,0].set_ylabel ('m')
    ax[0,0].legend(['Position - X'])

    ax[1,0].plot(dfname[colname3], c='#D95319')
    ax[1,0].set_ylabel ('m/s^2')
    ax[1,0].set_xlabel ('Time')
    ax[1,0].legend(['acc X'])

    ax[0,1].plot(dfname[colname2], c='#0072BD')
    ax[0,1].set_title(id + ' - ' + title + ' - Y ')
    ax[0,1].set_ylabel ('m')
    ax[0,1].legend(['Position - Y'])

    ax[1,1].plot(dfname[colname4], c='#D95319')
    ax[1,1].set_ylabel ('m/s^2')
    ax[1,1].set_xlabel ('Time')
    ax[1,1].legend(['acc Y'])

    plt.savefig("./DATA/Final_DataFrames_Figures/"+id + ' - ' +title)
    plt.show()

plot_final(final_df, 'Right_Knee_X', 'Right_Knee_Y', 'Pierna_acc_x', 'Pierna_acc_y', 'Right Knee')
plot_final(final_df, 'Right_Elbow_X', 'Right_Elbow_Y', 'Brazo_acc_x', 'Brazo_acc_y', 'Right Elbow')

# Add subject, repetition and timestamp columns
final_df.insert(0,'Subject', [subject for i in range(final_df.shape[0])])
final_df.insert(1,'Movimiento', [movement for i in range(final_df.shape[0])])
final_df.insert(2,'Repetition', [repetition for i in range(final_df.shape[0])])

final_df.to_csv("./DATA/Final_DataFrames/"+id+'.csv')
final_df