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

df_both = df_input.merge(df_fill, how = 'outer', left_index = True, right_index = True, suffixes = (None, '_y')).ffill(axis = 0)
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
n_features = len(train_scaled.columns)

train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = seq_size, batch_size = n_features)
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length = seq_size, batch_size = n_features)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN

model = Sequential([
    SimpleRNN(20, return_sequences = True, input_shape = (seq_size, n_features)),
    SimpleRNN(20, return_sequences = True),
    SimpleRNN(3)
])

model.summary()

history = model.fit_generator(train_generator, validation_data = test_generator,
                              epochs = 50, steps_per_epoch = 10)

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