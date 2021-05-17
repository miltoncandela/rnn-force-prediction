## Make a matplotlib plot from XY coordinates
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def coordsPlot(df, res, nombre, refresco = 0.1):
    columns = df.columns

    coordxV = []
    coordyV = []

    for col in columns:
        frame = df.iloc[0, :]
        especi = frame[col]
        especi = especi.strip('()\s')
        especi = especi.replace(' ', '')
        coordx, coordy, acc = especi.split(',')

        if ((float(coordx) != 0) or (float(coordy) != 0)):
            coordxV.append(float(coordx))
            coordyV.append(float(coordy))

    ph, = plt.plot(coordxV, coordyV, marker = 'o', color = 'red', linestyle = '')
    ax = plt.gca()
    ax.set_xlim([0, res[0]])
    ax.set_ylim([0, res[1]])
    ax.set_xlabel('Coordenadas en X (Pixeles)')
    ax.set_ylabel('Coordenadas en Y (Pixeles)')
    ax.set_title('Keypoints en 2D de ' + nombre)

    taza_ref = 0.1

    for i in range(1,len(df.index)):
        coordxV = []
        coordyV = []

        for col in columns:
            frame = df.iloc[i,:]
            especi = frame[col]
            especi = especi.strip('()\s')
            especi = especi.replace(' ', '')
            coordx, coordy, acc = especi.split(',')

            if((float(coordx) != 0) or (float(coordy) != 0)):

                coordxV.append(float(coordx))
                coordyV.append(res[1] - float(coordy) - 1)

        ph.set_xdata(coordxV)
        ph.set_ydata(coordyV)

        plt.pause(refresco)

estudiante_res = (1920, 1080)
voltereta_res = (1280, 720)
escalado_res = (299, 299)

dfEst = pd.read_csv('CoordenadasXY_TiroLibre.csv', index_col = 0)
dfVolt = pd.read_csv('CoordenadasXY_Voltereta.csv', index_col = 0)
dfVoltEsc = pd.read_csv('CoordenadasXY_Voltereta_Esc.csv', index_col = 0)
dfEstEsc = pd.read_csv('CoordenadasXY_Estudiante_Esc.csv', index_col = 0)

#coordsPlot(dfEst, estudiante_res, 'TirolibreEstudiante.mp4')
#coordsPlot(dfVolt, voltereta_res, 'Voltereta.mp4')
#coordsPlot(dfVoltEsc, escalado_res, 'Voltereta.mp4 (Escalado)')
coordsPlot(dfEstEsc, escalado_res, 'TiroLibreEstudiante.mp4 (Escalado)')