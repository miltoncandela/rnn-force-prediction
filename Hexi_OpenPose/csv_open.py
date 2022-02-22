import pandas as pd
import matplotlib.pyplot as plt


def get_coordinate(file_name, resolution, frame=0):
    df_normal = pd.read_csv('JSON_to_CSVs/XYCoordinates_' + file_name + '.csv', index_col=0)
    df_scaled = pd.read_csv('JSON_to_CSVs/XYCoordinates_' + file_name + '_Esc.csv', index_col=0)

    for i, parte in enumerate(df_normal.columns):
        x_normal, y_normal, acc = df_normal.iloc[frame, i].strip("()\s").replace(' ', '').split(',')
        x_scaled, y_scaled, acc = df_scaled.iloc[frame, i].strip("()\s").replace(' ', '').split(',')

        acc = round(float(acc) * 100, 2)

        if int(x_normal) == 0 and int(y_normal) == 0:
            x_scaled, y_scaled = 0, 0
        else:
            y_normal, y_scaled = int(resolution[1] - float(y_normal)), int(RESOLUTIONS['Rescaled'][1] - float(y_scaled))
        print("{} & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\".format(parte, x_normal, y_normal, x_scaled, y_scaled, acc, end='\n'))


TARGET_RES = (299, 299)
RESOLUTIONS = {'Student': (1920, 1080), 'Taekwondo': (1280, 720), 'Hexiwear': (640, 352), 'Rescaled': TARGET_RES,
               'Aime': (480, 848)}
get_coordinate('Taekwondo', RESOLUTIONS['Taekwondo'], frame=650)

def intel_coord(file_name, resolution, frame=0):
    body_intel = {'Nose': 0, 'Neck': 1, 'Right Shoulder': 2, 'Right Elbow': 3, 'Right Wrist': 4, 'Left Shoulder': 5,
                  'Left Elbow': 6, 'Left Wrist': 7, 'Right Hip': 8, 'Right Knee': 9, 'Right Ankle': 10, 'Left Hip': 11,
                  'Left Knee': 12, 'LAnkle': 13, 'Right Eye': 14, 'Left Eye': 15, 'Right Ear': 16, 'Left Ear': 17}

    df_intel = pd.read_csv(file_name)
    df_intel.columns = list(body_intel.keys())

    for parte in body_intel:
        x, y = df_intel.iloc[frame, body_intel[parte]][10:].strip("()\s").replace('x=', '').replace('y=', '').split(', ')
        x, y = float(x), float(y)

        x_scaled, y_scaled = int(x/resolution[0] * RESOLUTIONS['Rescaled'][0]),\
                             int(y/resolution[1] * RESOLUTIONS['Rescaled'][1])

        x, y = int(x), int(y)
        print('{} & {} & {} & {} & {} \\\\'.format(parte, x, y, x_scaled, y_scaled), end='\n')

# intel_coord('Aime-derecha-KeyPoints.csv', RESOLUTIONS['Aime'])
