## Make a matplotlib plot from XY coordinates
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

def coordsPlot(carpeta, res, refresco = 0.1):

    if(len(carpeta.split('_')) == 2):
        nombre = carpeta.split('_')[0] + '.mp4 (Escalado)'
    else:
        nombre = carpeta + '.mp4'
    camino = 'JSON_to_CSVs/CoordenadasXY_' + carpeta + '.csv'
    df = pd.read_csv(camino, index_col = 0)
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

#coordsPlot('Voltereta_Esc', escalado_res)

if __name__ == '__main__':
    proc1 = multiprocessing.Process(target = coordsPlot, args = ('Voltereta', voltereta_res))
    plt.plot()
    proc2 = multiprocessing.Process(target = coordsPlot, args = ('Voltereta_Esc', escalado_res))
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()