##
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('df_angulos.csv', index_col = 0)

print(df.isna().sum())
df = df.dropna()

estadisticas = df.describe().T

def norm(x):
    return ((x - estadisticas['mean']) / estadisticas['std'])

df_norm = norm(df)
labels = '1'

def crear_modelo():
    modelo = keras.Sequential([
        layers.Dense(64, activation = tf.nn.relu, input_shape = [len(df.keys())]),
        layers.Dense(64, activation = tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    modelo.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return (modelo)
modelo = crear_modelo()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end = '')

EPOCHS = 5000

history = modelo.fit(
    df_norm, labels, epochs = EPOCHS, validation_split = 0.2,
    verbose = 0, callbacks = [PrintDot()]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (MPG)')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ($MPG^2$)')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,20])
plot_history(history)

#early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
#history = modelo.fit(df_norm, labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])
#plot_history(history)

test_labels = 1

loss, mae, mse = modelo.evaluate(df_norm, test_labels, verbose = 0)

print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))

normed_test_data = 1
test_predictions = modelo.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values (MPG)')
plt.ylabel('Predictions (MPG)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error (MPG)')
_ = plt.ylabel('Count')
