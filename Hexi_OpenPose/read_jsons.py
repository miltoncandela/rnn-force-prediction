# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The current code reads all JSON files created from OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
# and joins them into a single DataFrame, which is then outputted into a CSV. It is worth noting that when the Computer
# Vision (CV) software does not detect a human joint, it defaults its value to 0, on the other hand, if no human is
# detected, the JSON file would be blank for that frame of time, although it would be taken as zeros for each joint.

import os
import json
import pandas as pd
import numpy as np

BODY_PARTS = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
              'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
              'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']


def json_to_csv(folder, current_resolution, target_resolution=None):
    """
    This function uses a folder that has multiple JSON files and joins them on a DataFrame, which is further outputted
    into a CSV on another path. The resolution of the source video must be provided, and it can be modified if the user
    would like to rescale the original file's resolution.

    :param string folder: Folder which has many JSON files, outputted by using inputting a video into OpenPose.
    :param tuple current_resolution: Video's original resolution.
    :param tuple target_resolution: Whether the pixels should be rescaled to a desired dimension.
    """

    # List all the JSON files and sets up variables whether the user has requested to rescale the coordinates
    path = 'output_JSONs/' + folder
    file_list = os.listdir(path)
    scaled = '' if target_resolution is None else '_Esc'
    target_resolution = current_resolution if target_resolution is None else target_resolution

    # Sets up a blank DataFrame onto which the coordinates would be appended
    csv_name = 'JSON_to_CSVs/XYCoordinates_' + folder + scaled + '.csv'
    df = pd.DataFrame(columns=BODY_PARTS)

    # The following loop reads out every JSON file that is found on the folder, and appends the found coordinates
    # onto the corresponding column. If the JSON established that OpenPose could not find a person on a given frame,
    # then that row would have zeros for each joint. Each datum in the DataFrame is a tuple that with the encoding:
    # (X_coordinate, Y_coordinate, estimated_accuracy) the coordinates are on pixels.

    for file in file_list:
        file_data = [json.loads(line) for line in open(path + '/' + file)]
        if len(file_data[0]['people']) == 0:
            key_points_data = np.zeros(len(BODY_PARTS))
        else:
            key_points = file_data[0]['people'][0]['pose_keypoints_2d']
            key_points_data = []
            for i in range(0, len(key_points), 3):
                joint_coord = (round(key_points[i]/current_resolution[0] * target_resolution[0]),
                               round(key_points[i + 1]/current_resolution[1] * target_resolution[1]), key_points[i + 2])
                key_points_data.append(joint_coord)
        df = df.append(dict(zip(BODY_PARTS, key_points_data)), ignore_index=True)
    df.index = range(df.shape[0])
    df.to_csv(path=csv_name)
    print('Saved CSV on {}, with data from {}'.format(csv_name, path))


RESOLUTIONS = {'Student': (1920, 1080), 'Taekwondo': (1280, 720), 'Hexiwear': (640, 352)}
TARGET_RES = (299, 299)

# The next for loop uses the previously described function to convert all the current folders which has JSON coordinates
# into a CSV, the dictionary contains the original resolution and it also rescales it into a (299, 299) resolution.

for current_folder in os.listdir('output_JSONs/'):
    curr_res = RESOLUTIONS[current_folder] if current_folder in ['Student', 'Taekwondo'] else RESOLUTIONS['Hexiwear']
    json_to_csv(current_folder, curr_res)
    json_to_csv(current_folder, curr_res, target_resolution=TARGET_RES)
