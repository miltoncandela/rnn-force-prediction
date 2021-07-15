import pandas as pd

#df = pd.read_csv('PublicRunBiomec/RBDS001processed.txt', sep = '\t', index_col = 'PercGcycle')
#print(df)

#pd.set_option('display.max_columns', None)

df_output = pd.read_csv('PublicRunBiomec/RBDS001runT25forces.txt', sep = '\t', index_col = 'Time').loc[:,['Fx', 'Fy', 'Fz']]
df_output.index = range(0, len(df_output.index))

df_input = pd.read_csv('PublicRunBiomec/RBDS001runT25markers.txt', sep = '\t')
df_input.index = [*range(0,len(df_output.index), 2)]
df_fill = pd.DataFrame(index = range(1, len(df_output.index), 2), columns = df_input.columns)

def class_cols(x):
    if x[-2:] == '_y':
        return(False)
    else:
        return(True)

df_both = df_input.merge(df_fill, how = 'outer', left_index = True,
                         right_index = True, suffixes = (None, '_y')).interpolate(method = 'linear', axis = 0)
columnas_rem = [class_cols(col) for col in df_both.columns]
df_input = df_both.loc[:,columnas_rem].drop('Time', axis = 1)

df_both = df_input.merge(df_output, how = 'inner', left_index = True, right_index = True)
print(df_both)

from sklearn.preprocessing import MinMaxScaler

p = 0.7
n = len(df_both.index)

train = df_both[:int(n * p)]
test = df_both[int(n * p):]

escalador = MinMaxScaler().fit(train)

columnas = train.columns

train_scaled = pd.DataFrame(escalador.transform(train), columns = columnas)
test_scaled = pd.DataFrame(escalador.transform(test), columns = columnas)

print(train_scaled)
print(test_scaled)

''''''
from keras.preprocessing.sequence import TimeseriesGenerator

seq_size = 5


train_output = train_scaled[['Fx', 'Fy', 'Fz']]
train_scaled.drop(['Fx', 'Fy', 'Fz'], inplace=True, axis = 1)

test_output = test_scaled[['Fx', 'Fy', 'Fz']]
test_scaled.drop(['Fx', 'Fy', 'Fz'], inplace=True, axis = 1)

n_features = len(train_scaled.columns)
print(train_scaled)
print(train_output)
import numpy as np
train_generator = TimeseriesGenerator(data = np.array(train_scaled), targets = np.array(train_output), length = seq_size, batch_size = n_features)
test_generator = TimeseriesGenerator(data = np.array(test_scaled), targets = np.array(test_output), length = seq_size, batch_size = n_features)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN, Input

model = Sequential()
model.add(Input(shape = (seq_size, n_features)))
model.add(SimpleRNN(100, return_sequences = True))
model.add(SimpleRNN(50))
model.add(Dense(3))

#optimizer = tf.keras.optimizers.RMSprop(0.01)
# opt = SGD(lr=0.01, momentum=0.9)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mse'])
model.summary()

history = model.fit(train_generator, validation_data = test_generator, epochs = 10)
# validation_data = (test_scaled, test_output), steps_per_epoch = 10
import matplotlib.pyplot as plt
def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'y', labels = 'Training loss')
    plt.plot(epochs, val_loss, 'r', labels = 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#plot_history(history)

from sklearn.metrics import r2_score

y_pred = model.predict(test_generator)
y_true = np.array(test_output)
#score = [r2_score(y_true[:,num],y_pred[:,num]) for num in [0,1,2]]
#print(score)
#print(np.mean(score))
print(y_pred)
print(y_true)
#print(escalador.inverse_transform(y_pred[:,0].reshape(1,-1)))
#print(escalador.inverse_transform(y_true[:,0].reshape(1,-1)))

for num in [0,1,2]:
    plt.plot(y_pred[:,num], 'y', label = 'Y Prediction')
    plt.plot(y_true[:,num], 'r', label = 'Y True')
    #plt.plot(y_pred[:,1], 'y', label = 'Y Prediction 2')
    #plt.plot(y_true[:,1], 'r', label = 'Y True 2')
    #plt.plot(y_pred[:,2], 'y', label = 'Y Prediction 3')
    #plt.plot(y_true[:,2], 'r', label = 'Y True 3')
    plt.title('Y pred and Y true ' + str(num + 1))
    plt.xlabel('Index')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
