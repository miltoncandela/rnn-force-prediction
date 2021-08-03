# Velocidad y aceleración respecto a frames
import pandas as pd
import numpy as np

df = pd.read_csv('JSON_to_CSVs/CoordenadasXY_Voltereta_Esc.csv', index_col = 0)

partes = df.columns

columnas_finales = []
for columna in partes:
    for coordenada in ['X', 'Y']:
        columnas_finales.append(columna + '_' + coordenada)

deltax = 1

velocidad = pd.DataFrame(data = np.ones(len(columnas_finales)).T, index = columnas_finales)
velocidad = velocidad.T

# Velocidad = dist/tiemp
def points_extract(row):
    return(map(float, row.strip('()\s').replace(' ', '').split(',')))

for row in range(1,len(df.index),deltax):
    velocidades = []
    for parte in range(len(partes)):

        px1, py1, _ =  points_extract(df.iloc[row - 1, parte])
        px2, py2, _ = points_extract(df.iloc[row, parte])
        distanciax = (px2 - px1)
        distanciay = (py2 - py1)

        #distancia = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)

        pixelesx = np.array([px1, px2])
        pixelesy = np.array([px1, px2])
        if (sum(pixelesx == 0) > 0):
            distanciax = np.nan
        if (sum(pixelesy == 0) > 0):
            distanciay = np.nan

        distancias = [distanciax, distanciay]
        velocidades.append(distancias[0])
        velocidades.append(distancias[1])
    velocidad = velocidad.append(dict(zip(columnas_finales, velocidades)), ignore_index=True)
velocidad.drop(0, axis = 0, inplace = True)
velocidad.index = range(0, len(velocidad.iloc[:,0]))

aceleracion = pd.DataFrame(data = np.ones(len(columnas_finales)).T,
                           index = columnas_finales)
aceleracion = aceleracion.T

for row in range(1,len(velocidad.index),deltax):
    aceleraciones = []
    for parte in range(len(partes)):
        px1, py1, _ = points_extract(df.iloc[row - 1, parte])
        px2, py2, _ = points_extract(df.iloc[row, parte])

        velx = (px2 - px1)
        vely = (py2 - py1)

        #distancia = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)

        pixelesx = np.array([px1, px2])
        pixelesy = np.array([px1, px2])
        if (sum(pixelesx == 0) > 0):
            velx = np.nan
        if (sum(pixelesy == 0) > 0):
            vely = np.nan

        velocidades = [velx, vely]

        aceleraciones.append(velocidades[0])
        aceleraciones.append(velocidades[1])
    aceleracion = aceleracion.append(dict(zip(columnas_finales, aceleraciones)), ignore_index=True)
aceleracion.drop(0, axis = 0, inplace = True)
aceleracion.index = range(0, len(aceleracion.iloc[:,0]))

print(df.head())
print('DF con velocidades con deltax = {}'.format(deltax))
print(velocidad.head())
print('DF con aceleraciones con deltax = {}'.format(deltax))
print(aceleracion.head())

print(velocidad.describe())
print(aceleracion.describe())