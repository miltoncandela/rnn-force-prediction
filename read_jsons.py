## Read all the created jsons

import os
import json
import pandas as pd
import numpy as np

BODY_PARTES = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
               'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
               'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
               'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
               'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

estudiante_res = (1920, 1080)
voltereta_res = (1280, 720)
hexi_res = (640, 352)

resx = 299
resy = 299

res_final = (resx, resy)

def from_JSON_to_CSV(carpeta, res_actual, escalado = True, res_objetivo = (299, 299)):

    camino = 'output_jsons/' + carpeta
    lista_archivos = os.listdir(camino)

    if escalado == False:
        res_objetivo = res_actual
        nombre_csv = 'JSON_to_CSVs/CoordenadasXY_' + carpeta + '.csv'
    else:
        nombre_csv = 'JSON_to_CSVs/CoordenadasXY_' + carpeta + '_Esc.csv'

    df = pd.DataFrame(data=np.ones(len(BODY_PARTES)).T, index=BODY_PARTES)
    df = df.T

    for i, archiv in enumerate(lista_archivos):
        data = []
        archivo = camino + '/' + archiv
        with open(archivo) as f:
            for line in f:
                data.append(json.loads(line))
            if len(data[0]['people']) == 0:
                continue
            keypoints = data[0]['people'][0]['pose_keypoints_2d']
            datos = []
            for i in range(0, round(len(keypoints)), 3):
                #tupla = '({}, {}, {})'.format(keypoints[i], keypoints[i + 1], keypoints[i + 2])
                tupla = (round(keypoints[i]/res_actual[0] * res_objetivo[0]), round(keypoints[i + 1]/res_actual[1] * res_objetivo[1]), keypoints[i + 2])
                datos.append(tupla)
        df = df.append(dict(zip(BODY_PARTES, datos)), ignore_index = True)
    df.drop(0, axis = 0, inplace = True)
    df.index = range(0, len(df.iloc[:,1]))

    print(nombre_csv)
    df.to_csv(nombre_csv)
    return(df)

print(from_JSON_to_CSV('Estudiante', estudiante_res, escalado = False))
#nombres = ['Mau1', 'Mau2', 'Mau3', 'Pers1', 'Pers2', 'Pers3', 'Rafa1', 'Rafa2', 'Rafa3']
#nombres = nombres[4:5]
#for name in nombres:
#    print(name)
#    print(from_JSON_to_CSV(name, hexi_res))