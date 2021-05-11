## Read all the created jsons

import os
import json
import pandas as pd
import numpy as np

lista_archivos = os.listdir('output_jsons')

BODY_PARTES = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
               'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
               'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
               'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
               'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

df = pd.DataFrame(data = np.ones(len(BODY_PARTES)).T, index = BODY_PARTES)
df = df.T

for i, archiv in enumerate(lista_archivos):
    data = []
    archivo = 'output_jsons/' + archiv
    with open(archivo) as f:
        for line in f:
            data.append(json.loads(line))
        if len(data[0]['people']) == 0:
            continue
        keypoints = data[0]['people'][0]['pose_keypoints_2d']
        datos = []
        for i in range(0, round(len(keypoints)), 3):
            #tupla = '({}, {}, {})'.format(keypoints[i], keypoints[i + 1], keypoints[i + 2])
            tupla = (keypoints[i], keypoints[i + 1], keypoints[i + 2])
            datos.append(tupla)
    df = df.append(dict(zip(BODY_PARTES, datos)), ignore_index = True)
df.drop(0, axis = 0, inplace = True)
df.index = range(0, len(df.iloc[:,1]))
print(df)