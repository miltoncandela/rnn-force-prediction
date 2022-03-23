# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code trains a Recurrent Neural Network (RNN) using sequential data from https://peerj.com/articles/3298/
# which contains runner's markers positions on the XYZ axis, these markers are attached to multiple joints on the body.
# For each run, a single XYZ force is extracted, and thus source (independent) variables would be the markers positions
# while the target (dependent) variables would be the forces, for each dimension (X, Y, Z).

# WARNING: Numpy version == 1.19.5, otherwise data could not be transformed into a generator and further into RNN:
# NotImplementedError: Cannot convert a symbolic Tensor (simple_rnn/strided_slice:0) to a numpy array.
# This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sys import exit
from random import seed, sample
import tensorflow as tf
available_devices = tf.config.experimental.list_physical_devices('GPU')
if len(available_devices) > 0:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        # tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import lfilter


def get_df(vel, sampling):
    """
    The following function joins multiple text files on the PublicRunBiomec folder, these files correspond to people
    having markers attached to their body, and so it was possible to track XYZ coordinates (markers), as well as
    forces exerted on these set of coordinates (forces). The subjects run on different velocities (2.5ms, 3.5ms, 4.5ms),
    and so files are labeled as "RBDS001runT25forces.txt", where 25 represent the velocity.

    :param string vel: Velocity from which the file will be gathered without using punctuations (25, 35, 45).
    :param string sampling: Type of sampling method that would be used to join the data, either up or down.
    :return pd.DataFrame: The pandas DataFrame corresponding to the concatenation of all files with that velocity.
    """

    # The function first list all the available files and subsets the ones corresponding to the force and markers,
    # this is a vital step because markers would be our source variables while forces would be our target variables.
    path = 'PublicRunBiomec/'
    file_list = os.listdir(path)
    markers, forces = ([bool(re.findall(vel + 'markers', file)) for file in file_list],
                       [bool(re.search(vel + 'forces', file)) for file in file_list])
    markers_idx, forces_idx = ([idx for idx, x in enumerate(markers) if x], [idx for idx, x in enumerate(forces) if x])

    # Once the file indices are being obtained, the next for loop iterates over the forces indices and concatenates
    # the current read file into a main DataFrame, which is called "df_forces".
    forces_columns = pd.read_csv(path + file_list[forces_idx[0]], nrows=0, sep='\t',
                                 index_col='Time').loc[:, ['Fx', 'Fy', 'Fz']].columns
    info_columns = list(forces_columns) + ['ID', 'Speed']
    df_forces = pd.DataFrame(columns=info_columns)
    for idx in forces_idx:
        df_temp = pd.read_csv(path + file_list[idx], sep='\t').loc[:, ['Fx', 'Fy', 'Fz']]
        n = 10
        b, a = [1 / n] * n, 1
        for force in ['Fx', 'Fy', 'Fz']:
            df_temp[force] = lfilter(b, a, df_temp[force])

        scaler = MinMaxScaler().fit(df_temp)
        scalers.append(scaler)
        df_temp = pd.DataFrame(scaler.transform(df_temp), columns=forces_columns)
        df_temp['ID'], df_temp['Speed'] = file_list[idx][4:7], file_list[idx][11:13]
        df_forces = pd.concat([df_forces, df_temp], ignore_index=True)
    df_forces = df_forces.reset_index(drop=True)
    df_forces['ID'], df_forces['Speed'] = pd.Categorical(df_forces.ID), pd.Categorical(df_forces.Speed)

    # A similar for loop is being implemented for the markers indices, although, the index of this DataFrame corresponds
    # to 2n, due to a difference in sampling frequency between forces and markers. And thus "space" needs to be
    # generated to implement an Up-Sampling technique and thus have the same granularity across both DataFrames.
    markers_columns = pd.read_csv(path + file_list[markers_idx[0]], nrows=0, sep='\t').columns
    df_markers = pd.DataFrame(columns=markers_columns)
    for idx in markers_idx:
        df_temp = pd.DataFrame(MinMaxScaler().fit_transform(pd.read_csv(path + file_list[idx], sep='\t')
                                                              .interpolate(method='linear', axis=0)
                                                              .reset_index(drop=True)),
                               columns=markers_columns)
        df_markers = pd.concat([df_markers, df_temp], ignore_index=True)
    df_markers = df_markers.dropna(axis=1, how='any')
    # Now, a "fill" DataFrame is being generated for each feature, this dataframe will create NANs for each in between
    # space that is not being covered by the "df_markers", which are (2n + 1) indices. After the merge is completed,
    # the columns from the fill DataFrame are removed and a linear interpolation method is being implemented on the
    # original DataFrame, which has a NAN value between each row. This Up-Sampling method makes sense, as it does not
    # remove valuable information, while it maintains the integrity of the data via a linear regression between rows.

    if sampling == 'up':
        df_markers.index = [*range(0, df_forces.shape[0], 2)]
        df_fill = pd.DataFrame(index=range(1, df_forces.shape[0], 2), columns=df_markers.columns)
        df_markers = df_markers.merge(df_fill, how='outer', left_index=True,
                                      right_index=True, suffixes=('_x', '_y')).interpolate(method='linear', axis=0)
        rem_columns = [False if col[-2:] == '_y' else True for col in df_markers.columns]
        df_markers = df_markers.loc[:, rem_columns].drop('Time_x', axis=1)
        df_markers.columns = [col[:-2] for col in df_markers]
        df_final = df_markers.merge(df_forces, how='inner', left_index=True, right_index=True)
    elif sampling == 'down':
        # df_forces = df_forces.drop(range(0, df_forces.shape[0], 2), axis=0).reset_index(drop=True)
        df_forces = df_forces.drop(range(0, df_forces.shape[0], 2), axis=0).reset_index(drop=True)
        df_final = pd.concat([df_markers, df_forces], axis=1)

    # Finally, as the granularity from both DataFrames coincide, it is possible to merge them via a inner join.
    return df_final


# Using the previous function, it concatenates all the velocities' DataFrames in a single DataFrame.
s_method = 'up'
scalers = []
df_both_25, df_both_35, df_both_45 = get_df('T25', s_method), get_df('T35', s_method), get_df('T45', s_method)

# df_both_25 = get_df('T25', s_method)

df_both = (pd.concat([df_both_25, df_both_35, df_both_45], ignore_index=True)
           .astype(np.float32).dropna(axis=1, how='any').reset_index(drop=True))
#df_both = (pd.concat([df_both_25], ignore_index=True)
#           .astype(np.float32).dropna(axis=1, how='any').reset_index(drop=True))
print(df_both.head())
print(df_both.shape)

train_ids = list(set(df_both.ID))
seed(200)
test_ids = sample(train_ids, 6)
seed(500)
valid_ids = sample(list(set(train_ids) - set(test_ids)), 3)
print(list(set(train_ids) - set(valid_ids + test_ids)), valid_ids, test_ids)
print(round(len(list(set(train_ids) - set(valid_ids + test_ids)))/len(train_ids) * 100, 2),
      round(len(valid_ids)/len(train_ids) * 100, 2), round(len(test_ids)/len(train_ids) * 100, 2))

# The whole dataset is divided to a training, validation and testing set. P_TEST defines the ratio from the dataset
# into the train_set and test dataset, while P_VALID defines the ratio from the train_set to training and validation.

train = df_both[~df_both.ID.isin(valid_ids + test_ids)].drop(['ID', 'Speed'], axis=1)
valid = df_both[df_both.ID.isin(valid_ids)].drop(['ID', 'Speed'], axis=1)
test = df_both[df_both.ID.isin(test_ids)].drop(['ID', 'Speed'], axis=1)

# Due to the use of Deep Learning, data must be normalized, and so MinMaxScaler would be used to fulfill this task,
# the only known information is the training dataset, and so the scaler would be fitted into this dataset and further
# used to transform both validation and testing dataset.

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# scaler = MinMaxScaler().fit(train)
# columns = train.columns

# train_scaled = pd.DataFrame(scaler.transform(train), columns=columns)
# valid_scaled = pd.DataFrame(scaler.transform(valid), columns=columns)
# test_scaled = pd.DataFrame(scaler.transform(test), columns=columns)

# Tokens de inicio y final para separar los ultimos valores de los sujetos en training, validation y testing.
BATCH_SIZE = 128
SEQ_SIZE = 5

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array


def df_to_generator(df_scaled):
    """
    This function takes a scaled DataFrame and separates the source and target variables into separate DataFrames, it
    also creates an object instance that represents both of the variables into a sequence of size SEQ_SIZE.

    :param pd.DataFrame df_scaled: Scaled DataFrame with source and target variables.
    :return (pd.DataFrame, pd.DataFrame, tf.keras.preprocessing.timeseries_dataset_from_array): Separated DataFrames
    depending on whether they have source or target variables, and a generator to train the RNN.
    """

    forces = ['Fx', 'Fy', 'Fz']
    df_output = df_scaled[forces]

    df_scaled.drop(forces, inplace=True, axis=1)

    n_features = df_scaled.shape[1]
    df_generator = TimeseriesGenerator(data=np.array(df_scaled), targets=np.array(df_output),
                                       length=SEQ_SIZE, batch_size=n_features)
    # df_generator = timeseries_dataset_from_array(data=np.array(df_scaled), targets=np.array(df_output),
    #                                              sequence_length=SEQ_SIZE)

    return df_scaled, df_output, df_generator


train_scaled, train_output, train_generator = df_to_generator(train)
valid_scaled, valid_output, valid_generator = df_to_generator(valid)
test_scaled, test_output, test_generator = df_to_generator(test)

for var in ['df_both_25', 'df_both_35', 'df_both_45', 'df_both',
            'train_ids', 'valid_ids', 'test_ids', 'train', 'valid', 'test']:
    exec(f'del {var}')

# Imports tensorflow library, which has deep learning function to build and train a Recurrent Neural Network, further
# code also sets up a GPU with 2GB as a virtual device for faster training, in case the user has one physical GPU.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

FOLDER = 'paper'
print(FOLDER)

def create_model(model_name=None):
    """
    Using tensorflow and keras, this function builds a sequential model with a RNN architecture, based on layers such
    as SimpleRNN, Dropout and a final Dense layer for the output. When it finishes training, the model is saved on the
    "saved_models" folder when the name parameter is different than None, the function also plots the metrics with
    respect to the number of epochs involve during the computation.

    :param string model_name: Name of the file on which the model would be saved.
    :return object: An already trained RNN, trained using the designated train_generator.
    """

    rnn = Sequential()
    # rnn.add(LSTM(8, input_shape=(SEQ_SIZE, N_FEATURES), activation='tanh',recurrent_activation='sigmoid',
    #              recurrent_dropout=0, unroll=False, use_bias=True))
    rnn.add(LSTM(8, input_shape=(SEQ_SIZE, N_FEATURES)))
    rnn.add(Dense(3))

    # Metrics:
    # MeanSquaredError
    # RootMeanSquaredError
    # MeanAbsoluteError
    # MeanAbsolutePercentageError
    # MeanSquaredLogarithmicError
    # CosineSimilarity
    # LogCoshError

    # tf.keras.losses.MeanSquaredError()
    rnn.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))
    history = rnn.fit(train_generator, validation_data=valid_generator, shuffle=False,
                      epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE)
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
    if model_name is not None:
        rnn.save('saved_models/{}/{}_E{}_S{}_B{}.h5'.format(FOLDER, model_name, EPOCH, SEQ_SIZE, BATCH_SIZE))

    hist = pd.DataFrame(history.history)
    hist.to_csv('figures/{}/history.csv'.format(FOLDER))
    hist['epoch'] = history.epoch

    for metric in METRICS.keys():
        metrics_fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel(METRICS[metric])
        plt.title('Training vs Validation {}'.format(metric.upper()))
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.plot(hist['epoch'], hist['val_' + metric], label='Validation')
        plt.legend()
        metrics_fig.savefig('figures/{}/{}_{}.png'.format(FOLDER, model_name, metric))

    return rnn


# The following chunks of code represents two ways a RNN model could be generated, either by CREATING or IMPORTING,
# please comment or uncomment the lines of code depending on the desired outcome.

N_FEATURES = train_scaled.shape[1]
METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)',
           'msle': 'Mean Squared Logarithmic Error (MSLE)'}
EPOCH = 25

# CREATING: Model generation via create_model, name is a parameter to save the model on "saved_models" folder.
name = 'mse_down'
model = create_model(name)

# IMPORTING: Model import via the load_model function, models are stored within the "saved_models" folder.
# from tensorflow.keras.models import load_model
# name = 'forward_using_target.h5'
# model = load_model('saved_models/' + name)

try:
    model.summary()
except NameError:
    exit('A model instance must be generated to continue, either by CREATING or IMPORTING')


def model_evaluation(predictions, true_values):
    """
    Manual evaluation of the model's predictions using R squared, pearson correlation and p-value. This is done by each
    dimension (X, Y, Z), and the function is called by each dataset (training, validation, testing).

    :param np.array predictions: Predicted forces using markers DataFrame and the trained RNN.
    :param np.array true_values: Original array of n rows by three columns (X, Y, Z) which contains forces.
    :return (list, float): A list of the three R squared scores depending on each dimension (X, Y, Z), and a rounded
    float which is the mean value of all coefficients of determination.
    """

    # predictions = np.concatenate((predictions, np.full((SEQ_SIZE - 1, 3), np.nan)))
    # predictions = np.array(pd.DataFrame({'F_x': predictions[:, 0], 'F_y': predictions[:, 1],
    #                                      'F_z': predictions[:, 2]}).fillna(method='ffill', axis=0))

    from sklearn.metrics import r2_score
    scores_r2 = [r2_score(true_values[SEQ_SIZE:, num], predictions[:, num]) for num in range(3)]

    from scipy.stats.stats import pearsonr
    scores_pearson = [np.abs(pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[0]) for num in range(3)]
    p_pearson = [pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[1] for num in range(3)]

    print('Pearson correlation:', scores_pearson)
    print('Pearson correlation (mean):', np.round(np.mean(scores_pearson), 4))
    print('P-value:', p_pearson)
    print('P-value (mean):', np.round(np.mean(p_pearson), 8))
    print('Coefficient of determination:', scores_r2)
    print('Coefficient of determination (mean):', np.round(np.mean(scores_r2), 4))
    return scores_r2, float(np.round(np.mean(scores_r2), 4))


print('*** Training evaluation ***')
train_scores, train_det = model_evaluation(model.predict(train_generator), np.array(train_output))
print('*** Validation evaluation ***')
valid_scores, valid_det = model_evaluation(model.predict(valid_generator), np.array(valid_output))
print('*** Testing evaluation ***')
test_scores, test_det = model_evaluation(model.predict(test_generator), np.array(test_output))
det_scores = [train_det, valid_det, test_det]
x_scores = [train_scores[0], valid_scores[0], test_scores[0]]
y_scores = [train_scores[1], valid_scores[1], test_scores[1]]
z_scores = [train_scores[2], valid_scores[2], test_scores[2]]

# The previously obtained scores, although they are printed on the console using the declared function model_evaluation,
# would be plotted using a bar plot. Each bar represents a force dimension and the set of bars represent each type of
# dataset available, and so a for loop that changes the bars position have to be employed to visually see the metrics.

bars = ('Training', 'Validation', 'Testing')
BAR_WIDTH = 0.15
y_pos = np.arange(len(bars))
names = ['F_x', 'F_y', 'F_z', 'Mean']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

r2_fig = plt.figure()
for i, score in enumerate([x_scores, y_scores, z_scores, det_scores]):
    plt.bar([x + BAR_WIDTH*i for x in y_pos], score, width=BAR_WIDTH, label=names[i], color=colors[i])
plt.title('Coefficient of determination (R Squared) across datasets ({} Epochs)'.format(EPOCH))
plt.ylabel('Coefficient of determination ($R^{2}$)')
plt.xticks([r + BAR_WIDTH*1.5 for r in y_pos], bars)
ax = plt.gca()
ax.set_ylim([0, 1])
plt.legend()
r2_fig.savefig('figures/{}/{}_r2_barplot.png'.format(FOLDER, name))

# Based on the procedure used on model_evaluation, the trained RNN would be used to predict the forces using the
# markers dataset, and so manually manipulate both the true values as well as the predicted values.

y_prediction = model.predict(test_generator)
# y_prediction = np.concatenate((y_prediction, np.full((SEQ_SIZE - 1, 3), np.nan)))
# y_prediction = np.array(pd.DataFrame({'F_x': y_prediction[:, 0], 'F_y': y_prediction[:, 1],
#                                       'F_z': y_prediction[:, 2]}).fillna(method='ffill', axis=0))
y_true = np.array(test_output)[SEQ_SIZE:, :]


def inv_scaler(y, current_scaler):
    """
    This function scales the data back to their original range and domain, although each sklearn scaler object has
    a .inverse_transform method, this method takes expects a DataFrame or Numpy Array with the same number of columns
    from which it was declared. However, it is not of our interests to scale back the source variables such as the
    markers, only real forces, and so we create dummy variables and extract only the desired target variables.

    :param np.array y: Scaled array using a sklearn scaler, such as MinMaxScaler().
    :param object current_scaler: sklearn.preprocessing object used to scale all the datasets.
    :return np.array: Inverse escalated array, which has original values (N).
    """
    df = pd.DataFrame(dict(zip(test_scaled.columns, np.ones(y.shape[0]))), index=range(y.shape[0]))
    df['Fx'], df['Fy'], df['Fz'] = y[:, 0], y[:, 1], y[:, 2]
    y_esc = current_scaler.inverse_transform(df)[:, -3:]
    return y_esc


#y_prediction_esc, y_true_esc = inv_scaler(y_prediction, scaler), inv_scaler(y_true, scaler)
y_prediction_esc, y_true_esc = y_prediction, y_true
pd.DataFrame(y_prediction).to_csv('figures/{}/y_pred.csv'.format(FOLDER))
pd.DataFrame(y_true).to_csv('figures/{}/y_true.csv'.format(FOLDER))

# Now automatic evaluation of the model takes place, the set of metrics used correspond to the ones that were tracked
# of the training and validation dataset, collected within the training of the RNN.

test_metrics = model.evaluate(test_generator, verbose=0)
print('*** Test metrics ***')
for i, test_metric in enumerate(['loss'] + list(METRICS.keys())):
    print(test_metric, test_metrics[i])


def plot_results(predictions, true_values):
    """
    Uses matplotlib to visually see whether a correlation is being observed, although a subset of the first 100 rows
    is used. As there are thousands of samples and it is nearly impossible to correctly visualize them on one plot.

    :param np.array predictions: Non-scaled predictions from the RNN.
    :param np.array true_values: Non-scaled real data from testing dataset.
    """

    for idx, force in enumerate(['Fx', 'Fy', 'Fz']):
        forces_fig = plt.figure()
        plt.plot(predictions[:100, idx], 'y', label='Predicted')
        plt.plot(true_values[:100, idx], 'r', label='True')
        plt.title('True and predicted force of {}'.format(force))
        plt.xlabel('Index')
        plt.ylabel('Force (N)')
        plt.legend()
        forces_fig.savefig('figures/{}/{}_predict_{}.png'.format(FOLDER, name, force))


plot_results(y_prediction_esc[:100, :], y_true_esc[:100, :])


'''
Code in development to real time plotting

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