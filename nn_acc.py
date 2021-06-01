# Input de velocidades y output de aceleraciones

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('JSON_to_CSVs/CoordenadasXY_Voltereta_Esc.csv', index_col = 0)

# Velocidad = dist/tiemp
def points_extract(row):
    return(row.strip('()\s').replace(' ', '').split(','))

def serieParte(df):
    s = pd.Series({'Pixeles' : 0})
    for column in df.columns:
        parte = df.loc[:,column]
        for axis in ['X', 'Y']:
            parteX = parte.apply(lambda x: float(points_extract(x)[0]))
            parteY = parte.apply(lambda x: float(points_extract(x)[1]))

            parteX[parteX == 0] = np.nan
            parteX = parteX.bfill(axis='rows')

            parteY[parteY == 0] = np.nan
            parteY = parteY.bfill(axis='rows')
        s = s.append(parteX, ignore_index=True)
        s = s.append(parteY, ignore_index=True)
    s.drop(s.index[0], inplace = True)
    s.index = range(0, len(s.index))
    return s

pixeles = serieParte(df)
pixeles.dropna(axis = 0, inplace = True)
pixeles.index = range(0, len(pixeles.index))
print('Serie pixeles')
print(pixeles)
print('Descripción de serie pixeles')
print(pixeles.describe())

def vel(serie):
    sVel = pd.Series([0])
    for row in range(1,len(serie.index)):
        #print(row)
        px1 = serie[row - 1]
        px2 = serie[row]
        #print(px1, px2)

        #sDes[row] = px2
        distancia = px2 - px1
        sVel[row] = distancia

        #print(sVel)
        #sVel.append(dict(zip('Nose_X', distanciax)))
    #sVel.drop(0, axis = 0, inplace= True)
    return(sVel)

def acc(serie):
    sAcc = pd.Series([0, 0])
    for row in range(1,len(serie.index)):

        px1 = serie[row - 1]
        px2 = serie[row]

        velocidadx = px2 - px1
        sAcc[row] = velocidadx
        #sVel.append(dict(zip('Nose_X', distanciax)))
    #sAcc.drop(0, axis = 0, inplace= True)
    return(sAcc)

sDes = pixeles
sVel = vel(pixeles)
print('Descripción de serie velocidades')
print(sVel.describe())
sAcc = acc(sVel)
print('Descripción de serie aceleraciones')
print(sAcc.describe())

#sDes1 = pd.Series(sNose_X[0])
#sDes1, sVel1 = vel(sNose_X, sDes1)
#sAcc1 = acc(sVel1)

frame = {'Des' : sDes,
         'Vel' : sVel}

X = pd.DataFrame(frame)
#X = np.array(sDes)
print('DataFram de pixeles y velocidades')
print(X)
print('Dimensiones')
print(X.shape)
Y = np.array(sAcc)
print('Numpy array de aceleraciones')
print(Y)
print('Dimensiones')
print(Y.shape)

import sklearn.model_selection as model_selection
X_train , X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size = 0.7, random_state = 101)

def crear_modelo():
    modelo = keras.Sequential([
        #layers.Flatten(input_shape = X_train.shape),
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(X.columns)]),
        layers.Dense(64, activation=tf.nn.softmax), layers.Dense(64, activation=tf.nn.sigmoid),
        layers.Dense(64, activation=tf.nn.tanh), layers.Dense(32, activation=tf.nn.tanh),
        layers.Dense(16, activation=tf.nn.tanh),
        layers.Dense(1)])

    optimizer = tf.keras.optimizers.RMSprop(0.01)
    #opt = SGD(lr=0.01, momentum=0.9)

    metrics = ['mae', 'mse']
    modelo.compile(loss = 'mean_squared_error',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return (modelo)
modelo = crear_modelo()

epoch = 1000
history = modelo.fit(
    X_train, y_train, epochs = epoch, validation_split = 0.2, verbose = 2)

predichos = modelo.predict(X_test)
#y_test = np.reshape(y_test, (388, 1))
y_test = np.reshape(y_test, predichos.shape)
print(predichos.shape)
#predichos = np.reshape(predichos, (388, 1))
print(predichos.shape)
print(y_test.shape)
y_test = np.round(y_test)
predichos = np.round(predichos)
s1 = pd.Series(list(y_test))
s2 = pd.Series(list(predichos))
frame = {'ref' : s1, 'pred' : s2}
df_pred = pd.DataFrame(frame)
print(df_pred)
print(np.round(np.sum(predichos == y_test)/len(y_test), 2))

import matplotlib.pyplot as plt
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (MPG)')
    plt.plot(hist['epoch'], hist['mae'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,5])
    plt.show()
plot_history(history)

test_predictions = modelo.predict(X_test).flatten()
print(sum(test_predictions == 0)/len(test_predictions))
plt.scatter(y_test, test_predictions, alpha = 0.4)
plt.xlabel('True Values (MPG)')
plt.ylabel('Predictions (MPG)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
'''

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error (MPG)')
_ = plt.ylabel('Count')
plt.show()



loss, _, mae, mse = modelo.evaluate(X_test, y_test, verbose = 2)
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(loss))
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mse))
'''