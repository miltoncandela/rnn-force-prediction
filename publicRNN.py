import pandas as pd
import numpy as np
import os
import re

#pd.set_option('display.max_columns', None)

def getdf(vel):
    camino = 'PublicRunBiomec/'
    lista_archivos = os.listdir(camino)
    marcadores = [bool(re.findall(vel + 'markers', archivo)) for archivo in lista_archivos]
    fuerzas = [bool(re.search(vel + 'forces', archivo)) for archivo in lista_archivos]
    idx_marcadores = [i for i, x in enumerate(marcadores) if x]
    idx_fuerzas = [i for i, x in enumerate(fuerzas) if x]

    df_output = pd.read_csv(camino + lista_archivos[idx_fuerzas[0]], sep = '\t', index_col = 'Time').loc[:,['Fx', 'Fy', 'Fz']]
    columnas_output = df_output.columns
    df_output = pd.DataFrame(columns = columnas_output)
    for idx in idx_fuerzas:
        df_temp = pd.read_csv(camino + lista_archivos[idx], sep = '\t', index_col = 'Time').loc[:,['Fx', 'Fy', 'Fz']]
        df_output = pd.concat([df_output, df_temp], ignore_index = True)
    df_output.index = range(len(df_output.index))

    df_input = pd.read_csv(camino + lista_archivos[idx_marcadores[0]], sep = '\t')
    columnas_input = df_input.columns
    df_input = pd.DataFrame(columns = columnas_input)
    for idx in idx_marcadores:
        df_temp = pd.read_csv(camino + lista_archivos[idx], sep = '\t')
        df_input = pd.concat([df_input, df_temp], ignore_index = True)
    df_input.index = [*range(0,len(df_output.index), 2)]

    df_fill = pd.DataFrame(index = range(1, len(df_output.index), 2), columns = df_input.columns)

    df_both = df_input.merge(df_fill, how = 'outer', left_index = True,
                             right_index = True, suffixes = (None, '_y')).interpolate(method = 'linear', axis=0)
    columnas_rem = [False if col[-2:] == '_y' else True for col in df_both.columns]
    df_input = df_both.loc[:, columnas_rem].drop('Time', axis = 1)

    df_both = df_input.merge(df_output, how = 'inner', left_index=True, right_index=True)
    return(df_both)
df_both_25, df_both_35, df_both_45 = getdf('T25'), getdf('T35'), getdf('T45')

df_both = pd.concat([df_both_25, df_both_35, df_both_45], ignore_index=True)
df_both.index = range(len(df_both.index))
print(df_both)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

p_test = 0.8
n = len(df_both.index)
train, test = df_both[:int(n * p_test)], df_both[int(n * p_test):]

p_valid = 0.7
n = len(train.index)
train, valid = train[:int(n * p_valid)], train[int(n * p_valid):]

escalador = MinMaxScaler().fit(train)
columnas = train.columns

train_scaled = pd.DataFrame(escalador.transform(train), columns = columnas)
valid_scaled = pd.DataFrame(escalador.transform(valid), columns = columnas)
test_scaled = pd.DataFrame(escalador.transform(test), columns = columnas)

batch_size = train_scaled.shape[0]/4

import matplotlib.pyplot as plt

'''
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

ph, = ax.plot(1, 1, 1, marker='o', color='red')
ax.set_xlabel('Coordenadas en X (Pixeles)')
ax.set_ylabel('Coordenadas en Y (Pixeles)')
ax.set_zlabel('Coordenadas en Z (Pixeles)')
#plt.show()


for point in test_scaled.index:
    for i in range(0, len(test_scaled.columns)//3):
        print(test_scaled.iloc[point,i:(i + 3)])
        X, Y, Z = test.iloc[point,i:(i + 3)]
        print(X, Y, Z)

        #ph.set_xdata(X)
        #ph.set_ydata(Y)
        #ph.set_zdata(Z)

        plt.pause(0.1)
        ph._offsets3d = (X, Y, Z)
        #plt.show()
        i = i + 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
x = np.random.normal(size=(80,3))
df = pd.DataFrame(x, columns=["x","y","z"])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
sc = ax.scatter([],[],[], c='darkblue', alpha=0.5)
def update(i):
    sc._offsets3d = (df.x.values[:i], df.y.values[:i], df.z.values[:i])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3)
ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(df), interval=70)
plt.tight_layout()
plt.show()
'''

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

seq_size = 5
def df_to_generator(df_scaled, seq_size = 5):
    df_output = df_scaled[['Fx', 'Fy', 'Fz']]
    df_scaled.drop(['Fx', 'Fy', 'Fz'], inplace = True, axis = 1)

    n_features = len(df_scaled.columns)
    #df_generator = TimeseriesGenerator(data = np.array(df_scaled), targets = np.array(df_output), length = seq_size, batch_size = n_features)
    df_generator = timeseries_dataset_from_array(data = np.array(df_scaled), targets = np.array(df_output), sequence_length = seq_size)
    return(df_scaled, df_output, df_generator)
train_scaled, train_output, train_generator = df_to_generator(train_scaled, seq_size)
valid_scaled, valid_output, valid_generator = df_to_generator(valid_scaled, seq_size)
test_scaled, test_output, test_generator = df_to_generator(test_scaled, seq_size)

n_features = len(train_scaled.columns)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(200, return_sequences = True, input_shape = (seq_size, n_features)))
model.add(Dropout(.8))
model.add(SimpleRNN(200, return_sequences = True))
model.add(Dropout(.8))
model.add(SimpleRNN(100))
model.add(Dropout(.6))
model.add(Dense(3))

model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), metrics = ['mae', 'mse', 'acc'])
model.summary()
history = model.fit(train_generator, validation_data = valid_generator, epochs = 125, verbose = 2, batch_size = batch_size)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.plot(hist['epoch'], hist['mse'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (MPG)')
    plt.plot(hist['epoch'], hist['mae'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['acc'], label = 'Train Accuracy')
    plt.plot(hist['epoch'], hist['val_acc'], label = 'Val Accuracy')
    plt.legend()
    plt.show()
plot_history(history)

y_pred = model.predict(test_generator)
y_true = np.array(test_output)

print('Dimensiones de la predicción:', y_pred.shape)
print('Dimensiones de la real:', y_true.shape)

#from sklearn.metrics import r2_score
#scores_r2 = [np.abs(r2_score(y_true[:,num],y_pred[:,num])) for num in range(3)]
#print('Coeficientes de determinación:', scores_r2)
#print('Coeficiente de determinación promedio:', np.mean(scores_r2))

def invEscalador(y, escalador):
    df = pd.DataFrame(dict(zip(test_scaled.columns, np.ones(y.shape[0]))), index=range(y.shape[0]))
    df['Fx'], df['Fy'], df['Fz'] = y[:, 0], y[:,1], y[:,2]
    y_esc = escalador.inverse_transform(df)[:, -3:]
    return(y_esc)
y_pred_esc, y_true_esc = invEscalador(y_pred, escalador), invEscalador(y_true, escalador)

def plot_results(y_pred, y_true):
    for i, fuerza in enumerate(['Fx', 'Fy', 'Fz']):
        plt.plot(y_pred[:,i], 'y', label = 'Predicted value')
        plt.plot(y_true[:,i], 'r', label = 'True value')
        plt.title('True and predicted force of {}'.format(fuerza))
        plt.xlabel('Index')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.show()
plot_results(y_pred_esc[:100,:], y_true_esc[:100,:])