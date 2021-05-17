## Read all the created jsons

import os
import json
import pandas as pd
import numpy as np

lista_archivos = os.listdir('output_jsons/estudiante/normal')

BODY_PARTES = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
               'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
               'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
               'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
               'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

estudiante_res = (1920, 1080)
voltereta_res = (1280, 720)

df = pd.DataFrame(data = np.ones(len(BODY_PARTES)).T, index = BODY_PARTES)
df = df.T

resx = 299
resy = 299

for i, archiv in enumerate(lista_archivos):
    data = []
    archivo = 'output_jsons/estudiante/normal/' + archiv
    with open(archivo) as f:
        for line in f:
            data.append(json.loads(line))
        if len(data[0]['people']) == 0:
            continue
        keypoints = data[0]['people'][0]['pose_keypoints_2d']
        datos = []
        for i in range(0, round(len(keypoints)), 3):
            #tupla = '({}, {}, {})'.format(keypoints[i], keypoints[i + 1], keypoints[i + 2])
            tupla = (keypoints[i]/estudiante_res[0] * resx, keypoints[i + 1]/estudiante_res[1] * resy, keypoints[i + 2])
            datos.append(tupla)
    df = df.append(dict(zip(BODY_PARTES, datos)), ignore_index = True)
df.drop(0, axis = 0, inplace = True)
df.index = range(0, len(df.iloc[:,1]))
print(df)

df.to_csv('CoordenadasXY_Estudiante_Esc.csv')