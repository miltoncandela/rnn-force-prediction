# Input de velocidades y output de aceleraciones

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('JSON_to_CSVs/CoordenadasXY_Voltereta_Esc.csv', index_col = 0)

nose = df.loc[:,'Nose']

# Velocidad = dist/tiemp
def points_extract(row):
    return(row.strip('()\s').replace(' ', '').split(','))

#nosex = np.array([0])
sNose_X = nose.apply(lambda x: float(points_extract(x)[0]))

print(round((sum(sNose_X == 0)/len(sNose_X)) * 100, 2), '%')
print(sNose_X.describe())

sNose_X[sNose_X == 0] = np.nan
sNose_X = sNose_X.bfill(axis = 'rows')

print(sNose_X.describe())

sVel = pd.Series([0])
sDes = pd.Series(sNose_X[0])
# Velocidad = dist/tiemp

for row in range(1,len(sNose_X.index)):
    px1 = sNose_X[row - 1]
    px2 = sNose_X[row]

    sDes[row] = px2
    distanciax = px2 - px1
    sVel[row] = distanciax
    #sVel.append(dict(zip('Nose_X', distanciax)))
#sVel.drop(0, axis = 0, inplace= True)
print(sVel)

sAcc = pd.Series([0, 0])
# Velocidad = dist/tiemp

for row in range(1,len(sVel.index)):
    px1 = sVel[row - 1]
    px2 = sVel[row]

    velocidadx = px2 - px1
    sAcc[row] = velocidadx
    #sVel.append(dict(zip('Nose_X', distanciax)))
#sAcc.drop(0, axis = 0, inplace= True)
print(sAcc)

frame = {'Des' : sDes, 'Vel' : sVel}

X = pd.DataFrame(frame)
#X = np.array(sDes)
print(X)
print(X.shape)
Y = np.array(sAcc)

import sklearn.model_selection as model_selection
X_train , X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size = 0.85, test_size = 0.15, random_state = 101)

def crear_modelo():
    modelo = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(X.columns)]),
        layers.Dense(32, activation = tf.nn.relu), layers.Dense(16, activation = tf.nn.relu),
        layers.Dense(64, activation = tf.nn.softmax), layers.Dense(32, activation = tf.nn.softmax),
        layers.Dense(64, activation = tf.nn.tanh), layers.Dense(32, activation = tf.nn.tanh),
        layers.Dense(16, activation = tf.nn.tanh), layers.Dense(8, activation = tf.nn.tanh),
        layers.Dense(1)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    modelo.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return (modelo)
modelo = crear_modelo()

epoch = 10000
history = modelo.fit(
    X_train, y_train, epochs = epoch, validation_split = 0.2, verbose = 2)

predichos = modelo.predict(X_test)
y_test = np.reshape(y_test, (388, 1))
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

plt.scatter(y_test, test_predictions, alpha = 0.4)
plt.xlabel('True Values (MPG)')
plt.ylabel('Predictions (MPG)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error (MPG)')
_ = plt.ylabel('Count')
plt.show()

loss, mae, mse = modelo.evaluate(X_test, y_test, verbose = 2)
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(loss))
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mse))
