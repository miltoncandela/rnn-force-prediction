# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The following code trains a Recurrent Neural Network (RNN) using sequential data from https://peerj.com/articles/3298/
# which contains runner's markers positions on the XYZ axis, these markers are attached to multiple joints on the body.
# For each run, a single XYZ force is extracted, and thus source (independent) variables would be the markers positions
# while the target (dependent) variables would be the forces, for each dimension (X, Y, Z).

# WARNING: Numpy version == 1.19.5, otherwise data could not be transformed into a generator and further into RNN:
# NotImplementedError: Cannot convert a symbolic Tensor (simple_rnn/strided_slice:0) to a numpy array.
# This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
# Tensorflow version == 2.5.0, otherwise
# "TypeError: Unable to convert function return value to a Python type! The signature was () -> handle

# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

import os
import numpy as np
import pandas as pd
from sys import exit
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Imports tensorflow library, which has deep learning function to build and train a Recurrent Neural Network, further
# code also sets up a GPU with 2GB as a virtual device for faster training, in case the user has one physical GPU.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# from tensorflow.keras.preprocessing import timeseries_dataset_from_array

available_devices = tf.config.experimental.list_physical_devices('GPU')
if len(available_devices) > 0:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])  # 2047 MB RTX 3060
        # tf.config.experimental.set_memory_growth(gpu, True)


# from tensorflow.keras.layers import SimpleRNN, GRU
# from sklearn.preprocessing import StandardScaler


def get_df():
    path = './Final_DataFrames/'
    file_list = os.listdir(path)
    l_columns = list(set(pd.read_csv(path + file_list[0], nrows=0).columns) - {'Subject', 'Movimiento',
                                                                               'Repetition', 'Datetime'})
    df_final = pd.DataFrame(columns=l_columns + ['ID'])

    for i_file in range(len(file_list)):
        df = pd.read_csv(path + file_list[i_file])
        # curr_id = df.Subject.astype(str) + '_' + df.Movimiento.astype(str) + '_' + df.Repetition.astype(str)
        curr_id = df.Subject.astype(str)
        df = pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['Subject', 'Movimiento', 'Repetition', 'Datetime'],
                                                               axis=1)), columns=l_columns)
        df['ID'] = curr_id
        df_final = pd.concat([df_final, df], axis=0, ignore_index=True)

    # df_final['ID'] = df_final.Subject.astype(str) + '_' + df_final.Movimiento.astype(str) +
    # '_' + df_final.Repetition.astype(str)
    # df_final = df_final.drop(['Subject', 'Movimiento','Repetition','Datetime'], axis=1)
    return df_final


df_both = get_df()
ids = list(set([int(id_n.split('_')[0]) for id_n in list(set(df_both.ID))]))


def create_and_test_model(df_ini, l_ids, combination_id):
    # The whole dataset is divided to a training, validation and testing set. P_TEST defines the ratio from the dataset
    # into the train_set and test dataset, while P_VALID defines the ratio from the train_set to training and validation

    # ids : [training, validation, testing]
    train = df_ini[~df_ini.ID.isin(l_ids[1] + l_ids[2])].drop(['ID'], axis=1).reset_index(drop=True)
    valid = df_ini[df_ini.ID.isin(l_ids[1])].drop(['ID'], axis=1).reset_index(drop=True)
    test = df_ini[df_ini.ID.isin(l_ids[2])].drop(['ID'], axis=1).reset_index(drop=True)

    def df_to_generator(df):
        """
        This function takes a scaled DataFrame and separates the source and target variables into separate DataFrames,
        it also creates an object instance that represents both of the variables into a sequence of size SEQ_SIZE.

        :param pd.DataFrame df: Scaled DataFrame with source and target variables.
        :return (pd.DataFrame, pd.DataFrame, tf.keras.preprocessing.timeseries_dataset_from_array): Separated DataFrames
        depending on whether they have source or target variables, and a generator to train the RNN.
        """

        acc = ['Pierna_acc_z', 'Pierna_acc_y', 'Pierna_acc_x', 'Brazo_acc_z', 'Brazo_acc_y', 'Brazo_acc_x']
        df_output = df[acc]

        df.drop(acc, inplace=True, axis=1)

        df_generator = TimeseriesGenerator(data=np.array(df).astype(np.float16),
                                           targets=np.array(df_output).astype(np.float16),
                                           length=SEQ_SIZE, batch_size=df.shape[1])
        # df_generator = timeseries_dataset_from_array(data=np.array(np.array(df)),
        #                                              targets=np.array(df_output), sequence_length=SEQ_SIZE)

        return df_output, df_generator

    train_output, train_generator = df_to_generator(train)
    valid_output, valid_generator = df_to_generator(valid)
    test_output, test_generator = df_to_generator(test)

    n_features = train.shape[1] - train_output.shape[1]

    for var in ['train', 'valid', 'test']:
        exec(f'del {var}')

    def create_model(model_name=None):
        """
        Using tensorflow and keras, this function builds a sequential model with an RNN architecture, based on layers
        such as SimpleRNN, Dropout and a final Dense layer for the output. When it finishes training, the model is saved
        on the "saved_models" folder when the name parameter is different than None, the function also plots the metrics
        with respect to the number of epochs involve during the computation.

        :param string model_name: Name of the file on which the model would be saved.
        :return object: An already trained RNN, trained using the designated train_generator.
        """

        rnn = Sequential()
        # rnn.add(LSTM(8, input_shape=(SEQ_SIZE, N_FEATURES), activation='tanh',recurrent_activation='sigmoid',
        #              recurrent_dropout=0, unroll=False, use_bias=True))
        rnn.add(LSTM(16, input_shape=(SEQ_SIZE, n_features)))
        rnn.add(Dense(6))

        # Metrics:
        # MeanSquaredError
        # RootMeanSquaredError
        # MeanAbsoluteError
        # MeanAbsolutePercentageError
        # MeanSquaredLogarithmicError
        # CosineSimilarity
        # LogCoshError

        # tf.keras.losses.MeanSquaredError()
        rnn.compile(loss=tf.keras.losses.MeanAbsoluteError(), metrics=METRICS.keys(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))
        history = rnn.fit(train_generator, validation_data=valid_generator, shuffle=False,
                          epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE)
        # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
        if model_name is not None:
            rnn.save('saved_models/{}/{}_E{}_S{}_B{}.h5'.format(FOLDER, model_name, EPOCH, SEQ_SIZE, BATCH_SIZE))

        hist = pd.DataFrame(history.history)
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

    # CREATING: Model generation via create_model, name is a parameter to save the model on "saved_models" folder.
    name = 'mae_{}'.format(combination_id)
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
        Manual evaluation of the model's predictions using R squared, pearson correlation and p-value. This is done by
        each dimension (X, Y, Z), and the function is called by each dataset (training, validation, testing).

        :param np.array predictions: Predicted forces using markers DataFrame and the trained RNN.
        :param np.array true_values: Original array of n rows by three columns (X, Y, Z) which contains forces.
        :return (list, float): A list of the three R squared scores depending on each dimension (X, Y, Z), and a rounded
        float which is the mean value of all coefficients of determination.
        """

        # predictions = np.concatenate((predictions, np.full((SEQ_SIZE - 1, 3), np.nan)))
        # predictions = np.array(pd.DataFrame({'F_x': predictions[:, 0], 'F_y': predictions[:, 1],
        #                                      'F_z': predictions[:, 2]}).fillna(method='ffill', axis=0))

        scores_r2 = [round(r2_score(true_values[SEQ_SIZE:, num], predictions[:, num]), 3) for num in range(6)]

        # scores_pearson = [np.abs(pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[0]) for num in range(6)]
        # p_pearson = [pearsonr(true_values[SEQ_SIZE:, num], predictions[:, num])[1] for num in range(6)]

        '''
        print('Pearson correlation:', scores_pearson)
        print('Pearson correlation (mean):', np.round(np.mean(scores_pearson), 4))
        print('P-value:', p_pearson)
        print('P-value (mean):', np.round(np.mean(p_pearson), 8))
        print('Coefficient of determination:', scores_r2)
        print('Coefficient of determination (mean):', np.round(np.mean(scores_r2), 4))
        '''
        return scores_r2, float(np.round(np.mean(scores_r2), 3))

    print('*** Training evaluation ***')
    train_scores, train_det = model_evaluation(model.predict(train_generator), np.array(train_output))
    print('*** Validation evaluation ***')
    valid_scores, valid_det = model_evaluation(model.predict(valid_generator), np.array(valid_output))
    print('*** Testing evaluation ***')
    test_scores, test_det = model_evaluation(model.predict(test_generator), np.array(test_output))
    z_leg_scores = [train_scores[0], valid_scores[0], test_scores[0]]
    y_leg_scores = [train_scores[1], valid_scores[1], test_scores[1]]
    x_leg_scores = [train_scores[2], valid_scores[2], test_scores[2]]
    z_arm_scores = [train_scores[3], valid_scores[3], test_scores[3]]
    y_arm_scores = [train_scores[4], valid_scores[4], test_scores[4]]
    x_arm_scores = [train_scores[5], valid_scores[5], test_scores[5]]
    det_scores = [train_det, valid_det, test_det]

    score_id_name = ['Z_Leg', 'Y_Leg', 'X_Leg', 'Z_Arm', 'Y_Arm', 'X_Arm']
    with open('results/{}/{}.txt'.format(FOLDER, name.split('_')[0]), 'a') as f:
        f.write("Combination #{}\n".format(combination_id))
        for i_write, score in enumerate([z_leg_scores, y_leg_scores, x_leg_scores,
                                         z_arm_scores, y_arm_scores, x_arm_scores]):
            f.write("{}: {}\n".format(score_id_name[i_write], score))
        f.write("\n")

    # The previously obtained scores, although they are printed on the console using the declared function
    # model_evaluation, would be plotted using a bar plot. Each bar represents a force dimension and the set of bars
    # represent each type of dataset available, and so a for loop that changes the bars position have to be employed
    # to visually see the metrics.

    bars = ('Training', 'Validation', 'Testing')
    y_pos = np.arange(len(bars))
    names = ['Leg_z', 'Leg_y', 'Leg_x', 'Arm_z', 'Arm_y', 'Arm_x', 'Mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    r2_fig = plt.figure()
    for i_s, score in enumerate(
            [z_leg_scores, y_leg_scores, x_leg_scores, z_arm_scores, y_arm_scores, x_arm_scores, det_scores]):
        plt.bar([x + BAR_WIDTH * i_s for x in y_pos], score, width=BAR_WIDTH, label=names[i_s], color=colors[i_s])

    plt.title('Coefficient of determination (R Squared) across datasets ({} Epochs)'.format(EPOCH))
    plt.ylabel('Coefficient of determination ($R^{2}$)')
    plt.xticks([r + BAR_WIDTH * 1.5 for r in y_pos], bars)
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

    # Now automatic evaluation of the model takes place, the set of metrics used correspond to the ones that were
    # tracked of the training and validation dataset, collected within the training of the RNN.

    test_metrics = model.evaluate(test_generator, verbose=0)
    print('*** Test metrics ***')
    for i_metric, test_metric in enumerate(['loss'] + list(METRICS.keys())):
        print(test_metric, test_metrics[i_metric])

    def plot_results(predictions, true_values):
        """
        Uses matplotlib to visually see whether a correlation is being observed, although a subset of the first 100 rows
        is used. As there are thousands of samples and it is nearly impossible to correctly visualize them on one plot.

        :param np.array predictions: Non-scaled predictions from the RNN.
        :param np.array true_values: Non-scaled real data from testing dataset.
        """

        for idx, force in enumerate(['Acc_Leg_Z', 'Acc_Leg_Y', 'Acc_Leg_X', 'Acc_Arm_Z', 'Acc_Arm_Y', 'Acc_Arm_X']):
            forces_fig = plt.figure()
            plt.plot(predictions[:100, idx], 'y', label='Predicted')
            plt.plot(true_values[:100, idx], 'r', label='True')
            plt.title('True and predicted force of {}'.format(force))
            plt.xlabel('Index')
            plt.ylabel('Acceleration (m/s^2)')
            plt.legend()
            forces_fig.savefig('figures/{}/{}_predict_{}.png'.format(FOLDER, name, force))

    plot_results(y_prediction[:100, :], y_true[:100, :])


k_fold = 5
BATCH_SIZE = 128
SEQ_SIZE = 10
FOLDER = 'Dacia2'
METRICS = {'mae': 'Mean Absolute Error (MAE)',
           'mse': 'Mean Squared Error (MSE)',
           'msle': 'Mean Squared Logarithmic Error (MSLE)'}
EPOCH = 350
BAR_WIDTH = 0.05

'''
for i in range(k_fold + 1):
    df_both = get_df()
    train_ids = list(set(df_both.ID))
    seed(i)
    test_ids = sample(train_ids, floor(len(train_ids)*0.25))
    seed(i * 10 + 12)
    valid_ids = sample(list(set(train_ids) - set(test_ids)), floor(len(train_ids)*0.25))
    # print(list(set(train_ids) - set(valid_ids + test_ids)), valid_ids, test_ids)
    # print(round(len(list(set(train_ids) - set(valid_ids + test_ids)))/len(train_ids) * 100, 2),
    #       round(len(valid_ids)/len(train_ids) * 100, 2), round(len(test_ids)/len(train_ids) * 100, 2))
    train_ids = list(set(train_ids) - set(valid_ids + test_ids))
    print(train_ids, valid_ids, test_ids)

    create_and_test_model(df_both, [train_ids, valid_ids, test_ids], i)
'''

# 11 12 14 15
# 11 12
# 11    14
# 11       15
#    12 14
#    12    15
#       14 15

combinations = [[11, 12, 14, 15], [11, 12, 15, 14],
                [11, 14, 12, 15], [11, 14, 15, 12],
                [11, 15, 12, 14], [11, 15, 14, 12],
                [12, 11, 14, 15], [12, 11, 15, 14],
                [12, 14, 11, 15], [12, 14, 15, 11],
                [12, 15, 11, 14], [12, 15, 14, 11],
                [14, 11, 12, 15], [14, 11, 15, 12],
                [14, 12, 11, 15], [14, 12, 15, 11],
                [14, 15, 11, 12], [14, 15, 12, 11],
                [15, 11, 12, 14], [15, 11, 14, 12],
                [15, 12, 11, 14], [15, 12, 14, 11],
                [15, 14, 11, 12], [15, 14, 12, 11]]


def string_to_list(s):
    temp_l = [0, s]
    del temp_l[0]
    return temp_l


for i, combination in enumerate(combinations):
    combination = [str(n) for n in combination]
    # train_ids, valid_ids, test_ids = [list(ids) if len(ids) == 0 else ids for ids in
    #                                  [combination[:2], combination[2], combination[3]]]
    train_ids, valid_ids, test_ids = [[c_id for c_id in ids] if type(ids) == type([0]) else string_to_list(ids)
                                      for ids in [combination[:2], combination[2], combination[3]]]
    print(train_ids, valid_ids, test_ids)

    create_and_test_model(df_both, [train_ids, valid_ids, test_ids], i)
