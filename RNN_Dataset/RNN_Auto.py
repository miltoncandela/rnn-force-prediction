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
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        # tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import lfilter

# Evaluation

from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr


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
        # scalers.append(scaler)
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


def pre_process(s_method, seed_train=200, seed_test=500):
    # Using the previous function, it concatenates all the velocities' DataFrames in a single DataFrame.
    df_both_25, df_both_35, df_both_45 = get_df('T25', s_method), get_df('T35', s_method), get_df('T45', s_method)

    # df_both_25 = get_df('T25', s_method)

    df_both = (pd.concat([df_both_25, df_both_35, df_both_45], ignore_index=True)
               .astype(np.float32).dropna(axis=1, how='any').reset_index(drop=True))
    # df_both = (pd.concat([df_both_25], ignore_index=True)
    #           .astype(np.float32).dropna(axis=1, how='any').reset_index(drop=True))
    print(df_both.head())
    print(df_both.shape)

    train_ids = list(set(df_both.ID))
    seed(seed_train)
    test_ids = sample(train_ids, 6)
    seed(seed_test)
    valid_ids = sample(list(set(train_ids) - set(test_ids)), 3)
    print(list(set(train_ids) - set(valid_ids + test_ids)), valid_ids, test_ids)
    print(round(len(list(set(train_ids) - set(valid_ids + test_ids)))/len(train_ids) * 100, 2),
          round(len(valid_ids)/len(train_ids) * 100, 2), round(len(test_ids)/len(train_ids) * 100, 2))

    # The whole dataset is divided to a training, validation and testing set. P_TEST defines the ratio from the dataset
    # into the train_set and test dataset, while P_VALID defines the ratio from the train_set to training and validation.

    train = df_both[~df_both.ID.isin(valid_ids + test_ids)].drop(['ID', 'Speed'], axis=1)
    valid = df_both[df_both.ID.isin(valid_ids)].drop(['ID', 'Speed'], axis=1)
    test = df_both[df_both.ID.isin(test_ids)].drop(['ID', 'Speed'], axis=1)

    return train, valid, test


sampling_method = 'up'
train, valid, test = pre_process(sampling_method)

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

# Imports tensorflow library, which has deep learning function to build and train a Recurrent Neural Network, further
# code also sets up a GPU with 2GB as a virtual device for faster training, in case the user has one physical GPU.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU


def create_model(model_type, loss_func, model_name=None):
    """
    Using tensorflow and keras, this function builds a sequential model with a RNN architecture, based on layers such
    as SimpleRNN, Dropout and a final Dense layer for the output. When it finishes training, the model is saved on the
    "saved_models" folder when the name parameter is different than None, the function also plots the metrics with
    respect to the number of epochs involve during the computation.

    :param string model_name: Name of the file on which the model would be saved.
    :return object: An already trained RNN, trained using the designated train_generator.
    """

    rnn = Sequential()

    layers_dict = {'LSTM': LSTM(8, input_shape=(SEQ_SIZE, N_FEATURES), activation='tanh',recurrent_activation='sigmoid',
                     recurrent_dropout=0, unroll=False, use_bias=True),
                   'GRU': GRU(8, input_shape=(SEQ_SIZE, N_FEATURES)),
                   'Simple': SimpleRNN(8, input_shape=(SEQ_SIZE, N_FEATURES))}

    rnn.add(layers_dict[model_type])
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

    loss_dict = {'MSE': tf.keras.losses.MeanSquaredError(),
                 'MAE': tf.keras.losses.MeanAbsoluteError(),
                 'MSLE': tf.keras.losses.MeanSquaredLogarithmicError()}
    #if loss_func == 'MSE':

    rnn.compile(loss=loss_dict[loss_func], metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))
    print(rnn.summary())
    history = rnn.fit(train_generator, validation_data=valid_generator, shuffle=False,
                      epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE)
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    # if model_name is not None:
    #     rnn.save('saved_models/{}/{}_E{}_S{}_B{}.h5'.format(FOLDER, model_name, EPOCH, SEQ_SIZE, BATCH_SIZE))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    for metric in list(METRICS.keys()) + ['loss']:
        metrics_fig = plt.figure()
        plt.xlabel('Epoch')
        y_label = METRICS[metric] if metric != 'loss' else METRICS[loss_function.lower()]
        plt.ylabel(y_label)
        plt.title('Training vs Validation {}'.format(metric.upper()))
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.plot(hist['epoch'], hist['val_' + metric], label='Validation')
        plt.legend()
        metrics_fig.savefig('{}/figures/{}_E-{}.png'.format(results_folder_name, encoded_name, metric.lower()))

    return rnn


# The following chunks of code represents two ways a RNN model could be generated, either by CREATING or IMPORTING,
# please comment or uncomment the lines of code depending on the desired outcome.

N_FEATURES = train_scaled.shape[1]
METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)',
           'msle': 'Mean Squared Logarithmic Error (MSLE)'}
EPOCH = 25


def evaluate_predictions(predictions, true_values, metric='r2'):
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

    if metric == 'R2':
        scores = [r2_score(true_values[SEQ_SIZE:, num], predictions[:, num]) for num in range(3)]
    elif metric == 'Pearson':
        scores = [np.abs(pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[0]) for num in range(3)]
    elif metric == 'PVal':
        scores = [pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[1] for num in range(3)]

    return scores, float(np.round(np.mean(scores), 4))


def evaluate_model(model):

    df_results = pd.DataFrame()

    for metric in ['R2', 'Pearson', 'PVal']:
        train_scores, train_det = evaluate_predictions(model.predict(train_generator), np.array(train_output), metric)
        valid_scores, valid_det = evaluate_predictions(model.predict(valid_generator), np.array(valid_output), metric)
        test_scores, test_det = evaluate_predictions(model.predict(test_generator), np.array(test_output), metric)

        det_scores = [train_det, valid_det, test_det]
        x_scores = [train_scores[0], valid_scores[0], test_scores[0]]
        y_scores = [train_scores[1], valid_scores[1], test_scores[1]]
        z_scores = [train_scores[2], valid_scores[2], test_scores[2]]

        df_temp = pd.DataFrame(dict(zip(['{}_X'.format(metric), '{}_Y'.format(metric), '{}_Z'.format(metric),
                                         '{}_Avg'.format(metric)], [x_scores, y_scores, z_scores, det_scores])))
        df_results = pd.concat([df_results, df_temp], axis=1)
    df_results.index = ['Training', 'Validation', 'Testing']
    df_results.to_csv('{}/csv/{}.csv'.format(results_folder_name, encoded_name))


results_folder_name = 'E-{}_U-8_B-{}_S-{}'.format(EPOCH, BATCH_SIZE, SEQ_SIZE)

if not os.path.exists(results_folder_name):
    os.mkdir(results_folder_name)
    os.mkdir(results_folder_name + '/figures')
    os.mkdir(results_folder_name + '/csv')

for loss_function in ['MAE', 'MSLE']:
    for model_type in ['LSTM', 'GRU', 'Simple']:
        print('Processing L: {}, M: {}'.format(loss_function, model_type))
        encoded_name = 'S-{}_L-{}_M-{}'.format(sampling_method, loss_function.lower(), model_type.lower())
        evaluate_model(create_model(model_type, loss_function))
        print('Finish L: {}, M: {}'.format(loss_function, model_type))
