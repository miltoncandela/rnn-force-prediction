## Calcular los ángulos

##
#Trunk	The angle of a straight line connecting "Neck" and "MidHip" relative to vertical
#Hip	The angle of a straight line connecting "RHip" and "RKnee" relative to a straight line connecting "Neck" and "MidHip"
#Knee	The angle of "RHip", "RKnee" and "RAnkle"
#Ankle	The angle of a straight line connecting "RHeel" and "RSmallToe" relative to a straight line connecting "RKnee" and "RAnkle"
##

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('CoordenadasXY_Voltereta.csv', index_col = 0)

angulos = {'Trunk' : [['Neck', 'MidHip'], ['Nose','MidHip']],
           'HipR' : [['RHip', 'RKnee'], ['Neck', 'MidHip']],
           'AnkleR' : [['RHeel', 'RSmallToe'], ['RKnee', 'RAnkle']],
           'HipL': [['LHip', 'LKnee'], ['Neck', 'MidHip']],
           'AnkleL': [['LHeel', 'LSmallToe'], ['LKnee', 'LAnkle']],
           'KneeR' : ['RHip', 'RKnee', 'RAnkle'],
           'KneeL' : ['LHip', 'LKnee', 'LAnkle']}

df_angs = pd.DataFrame(data = np.ones(len(angulos.keys())), index = angulos.keys())
df_angs = df_angs.T

df_prob = df

def points_extract(row):
    return(map(float, row.strip('()\s').replace(' ', '').split(',')))

def get_ang(u, v):
    num = np.dot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if (num == 0 or den == 0):
        # print(frame, num, den)
        return(np.NAN)
    else:
        return(np.arccos(num / den) * (180 / np.pi))

for frame in range(0, len(df.index) - 1):
    angs = []
    for key in angulos.keys():
        columnas = angulos[key]
        if(len(columnas) == 2):
            df_temp = df_prob.iloc[frame,:][columnas[0] + columnas[1]]

            px1, py1, _ = points_extract(df_temp[0])
            px2, py2, _ = points_extract(df_temp[1])
            u = [px2 - px1, py2 - py1]

            px3, py3, _ = points_extract(df_temp[2])
            px4, py4, _ = points_extract(df_temp[3])
            v = [px4 - px3, py4 - py3]

            pixeles = np.array([px1, py1, px2, py2, px3, py3, px4, py4])

            ang = get_ang(u, v)
            if(sum(pixeles == 0) > 0):
                ang = np.nan
            angs.append(ang)
        elif(len(columnas) == 3):
            df_temp = df_prob.iloc[frame, :][columnas]
            #print(df_temp)
            px1, py1, _ = points_extract(df_temp[0])
            pxm, pym, _ = points_extract(df_temp[1])
            px2, py2, _ = points_extract(df_temp[2])

            u = [px1 - pxm, py1 - pym]
            v = [px2 - pxm, py2 - pym]

            pixeles = np.array([px1, py1, px2, py2, pxm, pym])

            ang = get_ang(u, v)
            if(sum(pixeles == 0) > 0):
                ang = np.nan
                #print(ang)
            angs.append(ang)
    df_angs = df_angs.append(dict(zip(angulos.keys(), angs)), ignore_index=True)
df_angs.drop(0, axis = 0, inplace = True)
df_angs.index = range(0, len(df_angs.iloc[:,0]))

funciones = [np.max, np.min, np.mean, np.median]
print(df_angs.agg(dict(zip(df_angs.columns, [funciones] * len(df_angs.columns)))))

# cos(theta) = (u * v)/(mu * mv)
#num = ux * vx + uy * vy
#print(ux * vx + uy * vy, np.dot([ux, uy], [vx, vy]))

#den = np.sqrt(ux ** 2 + uy ** 2) * np.sqrt(vx ** 2 + vy ** 2)
#print(np.sqrt(ux ** 2 + uy ** 2), np.linalg.norm([ux, uy]))