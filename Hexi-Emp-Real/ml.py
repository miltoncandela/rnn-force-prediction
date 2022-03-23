# Get Fatigue #
import pandas as pd

# EEG Analysis #
import os
import numpy as np
# import pandas as pd
from math import floor
from datetime import datetime
# from scipy.stats import stats
from collections import Counter
from sklearn.impute import KNNImputer

# Comb Features #
import copy
# import numpy as np
# import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create Model #
# import numpy as np
# import pandas as pd
from pickle import dump
from statistics import mean
import matplotlib.pyplot as plt
# from collections import Counter
# from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report


def get_fatigue():
    """
    The following function read a .CSV "Copia de Cuestionario FAS.csv" which is generated from a Google Forms with
    the FAS questions and each subject answers. It is worth noting that the name of some subjects was changed, as
    duplicates occurred on the first name, for these cases, the middle name was taking into consideration and joined
    via UpperCammelCase. A separate 'ids.txt' assigns the subjects ID depending on their name, this is important
    to join their scores to their biometric data, which is ID-encoded.

    :return pd.DataFrame: A DataFrame with subject IDs, their fatigue score "FAS", and their encoded Fatigue Score
    in 2-Class and 3-Class.
    """

    # According to each multiple choice answer, a score is assigned to each question.
    encoding_fatigue = {'Para nada': 1, 'Poco': 2, 'Regular': 3,
                        'Muy cierto': 4, 'Totalmente': 5}

    # The previous encoding for a numerical score, is reversed in some questions.
    encoding_fatigue_reversed = {'Para nada': 5, 'Poco': 4, 'Regular': 3,
                                 'Muy cierto': 2, 'Totalmente': 1}

    # CSV is read and information columns are removed (i.e. date).
    df = pd.read_csv('Empatica-Project-ALAS-main/Files/Copia de Cuestionario FAS.csv')
    df = df.iloc[:, [2] + [x for x in range(6, 16)]]

    # Only the first name is taken into account and set as the DataFrame index.
    df['Nombre completo'] = [nombre.split(' ')[0] for nombre in df['Nombre completo']]
    df = df.set_index('Nombre completo')
    df_encoded = pd.DataFrame(index=df.index)

    # A for loop transforms each question's answer to a numerical number of fatigue, inverting the score in questions
    # 4 and 10 (according to the FAS questionnaire), the total number of questions is 10.
    for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        current_encoding = encoding_fatigue
        if number in [4, 10]:
            current_encoding = encoding_fatigue_reversed
        df_encoded['Preg{}'.format(number)] = df.iloc[:, number - 1].map(current_encoding)

    # Each numerical value on each column is summed, and so a score is obtained for each subject.
    df_encoded = df_encoded.sum(axis=1).reset_index()
    df_encoded.columns = ['Name', 'FAS']

    # Afterwards, the ID for each subject name is read, using a .txt file called "ids.txt".
    ids_to_name = pd.read_csv('ids.txt', sep='\t', header=None).drop(1, axis=1)
    ids_to_name.columns = ['Name', 'Subject']

    # Both DataFrames are merged using "Name" as the primary key, and so each subject ID receives a fatigue score.
    df_encoded = pd.merge(df_encoded, ids_to_name, on='Name')

    # Categorical features are added with respect to each ID's score, including a 2-class and 3-class category.
    df_encoded['FAS_Cat_2'] = [1 if fatigue >= 22 else 0 for fatigue in df_encoded.FAS]
    df_encoded['FAS_Cat_3'] = [1 if (35 > fatigue >= 22) else 0 if fatigue < 22 else 2 for fatigue in df_encoded.FAS]

    # The subject is sorted and thus this DataFrame is returned by the function, having both ID and FAS score.
    df_encoded = df_encoded.sort_values('Subject', ascending=True).reset_index(drop=True)

    return df_encoded


def eeg_analysis():
    """
    The following function uses the pd.DataFrame that relates subject ID and their score of Fatigue, obtained via the
    get_fatigue() function. In addition, it adds EEG and Empatica features according to the raw files found in each ID
    folder, these devices have different granularity due to the different sampling rate, and so under-sampling was used.

    EEG band powers were obtained for each 2 seconds, while Empatica features were obtained in milli-seconds, so a set
    of statistical functions were applied to the Empatica data each 2 second interval, and so match the EEG sampling
    rate, resulting in a DataFrame with the same granularity.

    :return pd.DataFrame: A DataFrame with EEG band powers and Empatica's features according to the statistical methods
    and whether the global variable "Empatica" is true, else only EEG band powers would be returned. Both of these
    options have information features such as subject's ID, Second, Repetition and FAS score.
    """

    # The fatigue DataFrame is obtained via the function get_fatigue(), name is dropped as well as categorical features
    # such as the class, due to the fact that a regression machine learning model would be created.
    df_fatigue = get_fatigue().drop(['Name', 'FAS_Cat_2', 'FAS_Cat_3'], axis=1)

    # Function of the coefficient of variation.
    def cv(x): return np.std(x, ddof=1) / np.mean(x) * 100

    # The set of statistical functions is defined in agg_funcs list, as well as each function's name.
    agg_funcs = [np.mean, np.std, np.median, np.max, np.min, stats.iqr, stats.kurtosis, stats.skew, cv]
    funcs_names = ['Mean', 'StandardDeviation', 'Median', 'Maximum', 'Minimum', 'InterquartileRange', 'Kurtosis',
                   'Skewness', 'CoefficientVariation']

    # An empty DataFrame is created, on which all the subject's biometric data would be included, as well as features
    # that would identify their data: ID, Repetition, FAS score.
    df_total = pd.DataFrame()

    # The following for loop iterates over all sub-folders in "PruebasConBeeps", where each sub-folder corresponds to a
    # test identified by the following encoding: S**R**_ddmmyyyy_hhmm. Where the first asterisks represent the subject's
    # ID, while the second pair of asterisks corresponds to the repetition number. The hour is in 24-hour format, and
    # 0s were padded on IDs and Repetitions < 10 (i.e. ID = 9 would be converted to ID = 09 in the folder).
    for folder in os.listdir('PruebasConBeeps/'):
        # The following boolean flag is set to False as default, as it would determine if a subject's biometric data
        # would be used. In this case, EEG data was considered as the primary source of data, and so this data must
        # be complete in order to determine if a test would be taken into consideration for the model.
        aceptado = False

        # For the current tests, EEG data did not had timestamps, and so separate tests made by ID 99 were made
        # to obtain the most probable timestamp. This had two outcomes: EEG DataFrames with rows = 323 or = 325,
        # depending on the number of rows in the DataFrame, is the timestamp index that would be assigned, this
        # differs because the device may started getting data in second 2 or 3.
        index_eeg = {323: (pd.read_csv('PruebasConBeeps/S99R01_09102021_1407/Raw/Alpha.csv', index_col=0)
                           .apply(pd.to_numeric, errors='coerce').dropna(axis=0).index),
                     325: (pd.read_csv('PruebasConBeeps/S99R02_09102021_1414/Raw/Alpha.csv', index_col=0)
                           .apply(pd.to_numeric, errors='coerce').dropna(axis=0).index)}

        # An empty temporal DataFrame is created for each device, on which data would be included, in addition, the
        # variable n is initialized with a value of 0, this variable would be used to know EEG's number of rows.
        df_eeg = pd.DataFrame()
        df_emp = pd.DataFrame()
        n = 0

        # The following for loop would iterate over all the files within the "Raw" folder that is found on each
        # test sub-folder, this raw folder has EEG and Empatica CSV files, but Empatica Raw files start with "file".
        for df_name in os.listdir('PruebasConBeeps/{}/Raw/'.format(folder)):

            # Three conditions were applied to determine whether a file were EEG base band powers:
            # 1) If it ends with .csv.
            # 2) If it does not start with "file" (only empatica's files start this way).
            # 3) If it does not have underscore, as only ratio band powers have underscore.
            if df_name[-4:] == '.csv' and df_name[:4] != 'file' and len(df_name.split('_')) == 1:

                # The file name is simplified, as the ".csv" last part is removed.
                df_name = df_name[:-4]

                # The raw data is read into a DataFrame and their number of rows is stored into the n variable.
                df_raw = pd.read_csv('PruebasConBeeps/{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
                n = df_raw.shape[0]

                # Only complete EEG data is accepted, and so their number of rows must be either 323 or 325.
                if n in [323, 325]:
                    # If the clause is true, then the accepted flag is set to True, which would then be able to read
                    # empatica data, because the EEG data from the current test is complete.
                    aceptado = True
                    df_eeg_temp = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)

                    # The temporal DataFrame containing EEG's band powers is concatenated with the main EEG DataFrame.
                    df_eeg = pd.concat([df_eeg, df_eeg_temp], axis=1)

            # Five conditions were applied to determine whether an Empatica's CSV would be used:
            # 1) If it starts with "file".
            # 2) If it does have "ACC" or "IBI" following file, as these Empatica's features would not be used.
            # 3) If "aceptado" boolean feature is set to true, that is when EEG's dataset is complete.
            # 4) If "empatica" boolean feature is set to true, which determines if empatica's features would be used.
            if df_name[:4] == 'file' and df_name[4:7] != 'ACC' and df_name[4:7] != 'IBI' and aceptado and empatica:

                # The following statistical methods are currently not being applied:
                # quartiles, mean-of-squared values, range, slope, first derivative, second derivative,
                # ratio of maximum to minimum values, sum of absolute values, mean of absolute values,
                # autoregressive parameters, wavelet decomposition coefficients, signal-to-noise ratio (SNR),
                # correlations between biosignals, mean normalized frequency, power, magnitude of frequency response,
                # and Fourier transform.

                # On the other hand, the following statistical methods are currently being used:
                # mean, sd, median, maximum, minimum, interquartile range, skewness, kurtosis, coefficient of variation

                # The following set of lines of codes transforms the general Datetime from Empatica's features and
                # transforms it to relative seconds when initializing the test.
                df = pd.read_csv('PruebasConBeeps/{}/Raw/{}'.format(folder, df_name))
                df['Datetime'] = pd.to_datetime(df.Datetime).apply(datetime.timestamp)
                df['Datetime'] = df.Datetime - df.loc[0, 'Datetime']
                df['Second'] = df.Datetime.apply(floor)

                # Seconds are used to generate a set of bins, with size equal to the number of observations found in the
                # EEG dataset for the current subject and repetition. It is worth noting that at least one EEG file
                # must be read before an Empatica's file is read, in order to determine the number of observations on
                # which the dataset would be divided (number of bins).
                df['Bins'] = pd.cut(df['Second'], bins=[0] + list(index_eeg[n]), duplicates='drop', include_lowest=True)
                df = df.drop(['Second', 'Datetime'], axis=1).groupby('Bins')

                # The following if clause determines if the file is Temperature or not, because Temperature is a
                # feature that did not had many changes in their value over time, and so only mean would be used.
                if df_name.split('.')[0][4:] != 'Temp':
                    df = df.agg(agg_funcs)['value{}'.format(df_name.split('.')[0][4:])].reset_index(drop=True)
                    df.columns = [df_name.split('.')[0][4:] + '_' + func_name for func_name in funcs_names]
                # Otherwise, the set of statistical functions would be applied, with the name according to the desired
                # name declared before in the list "func_name".
                else:
                    df = df.agg([np.mean])['value{}'.format(df_name.split('.')[0][4:])].reset_index(drop=True)
                    df.columns = [df_name.split('.')[0][4:] + '_' + func_name for func_name in ['Mean']]

                # NULL values inside the dataset represent malfunctioning of the device during the test, and so a
                # polinomial interpolation of order 2 (cuadratic) would be applied to NA values between rows.
                df = df.replace(0, np.nan).interpolate(method='polynomial', order=2, axis=0,
                                                       limit_direction='forward', limit_area='inside')

                # On the other hand, NULL values outside the rows represent malfunctioning of the device before and
                # after the test, and so a KNN imputer would be used for those rows in the "outside" area.
                df = pd.DataFrame(KNNImputer(n_neighbors=4).fit_transform(np.array(df)), columns=df.columns)

                # The temporal DataFrame with one features is concatenated to the empatica's DataFrame for the subject.
                df_emp = pd.concat([df_emp, df], axis=1)

        # If the subject is accepted, then information features are included in the EEG DataFrame.
        if aceptado:
            df_eeg['Subject'], df_eeg['Repetition'], df_eeg['Second'] = folder[1:3], folder[4:6], index_eeg[n]
            # If empatica's boolean variable is set to true, then Empatica's DataFrame would be merged with the EEG.
            if empatica and n in [323, 325]:
                df_emp['Second'] = list(set(list(index_eeg[n])))
                df_eeg = pd.merge(df_eeg, df_emp, on='Second')
            # The EEG DataFrame, which now has features from both devices, then would be concatenated to a final
            # DataFrame for the current test that is being addressed by the main for loop.
            df_total = pd.concat([df_total, df_eeg], ignore_index=True)

        print(folder, aceptado, n)

    # Subject columns are transformed to a numerical feature.
    df_total['Subject'] = pd.to_numeric(df_total.Subject)
    df_fatigue['Subject'] = pd.to_numeric(df_fatigue.Subject)

    # Both fatigue features and biometric features are joined using Subject's ID as the primary key.
    df_total = pd.merge(df_total, df_fatigue, on='Subject')

    # Subject ID 99 is removed because it is only used for testing, on the other hand, subject ID 19 is removed due to
    # pain medication, which could disturb biometrics.
    df_total = df_total.drop(df_total[df_total.Subject.isin(['99', '19', 99, 19])].index, axis=0)

    print(len(df_total.Subject.unique()), df_total.Subject.unique())
    print(Counter([1 if (x >= 22) else 0 for x in
                   [df_total[df_total.Subject == sub].head(1).FAS.item() for sub in df_total.Subject.unique()]]))
    print(Counter([1 if (35 > x >= 22) else 0 if x < 22 else 2 for x in
                   [df_total[df_total.Subject == sub].head(1).FAS.item() for sub in df_total.Subject.unique()]]))

    return df_total


def comb_features():
    """
    The following function pre-process the DataFrame of EEG and ECG features with a set of three functions that:
    Combines features, remove outliers and normalizes data, according to variables defined in main as global variables.

    :return pd.DataFrame: Transformed DataFrame, ready to be fitted by a Machine Learning model and so make predictions.
    """
    def combined_features(df, information_features_combined):
        """
        The following functions takes a DataFrame with both target and source variables, the name of the target and
        information features is included in the list "information_features_combined", because they must be removed
        before doing the combinations between the features available, and so a DataFrame with the original features,
        as well as additional combined features, is returned.

        :param pd.DataFrame df: DataFrame with both source and target variables.
        :param list information_features_combined: List of information and target variables.
        :return pd.DataFrame: A DataFrame with combined features,
        """

        # A separate DataFrame is created to save the information features such as "Second", "ID".
        df_information_combined = df[information_features_combined]
        df = df.drop(information_features_combined, axis=1)
        df_combined = copy.deepcopy(df)

        epsilon = 0.000001
        names = list(df_combined.columns)
        combinations = []

        # The following for loop creates a set of combined features based on the source variables that are available.
        # It iterates over all the features on a separate DataFrame, and it applies a function. The result is further
        # saved on a column with the following encoding:

        # Name_i-I : Inverse on ith feature
        # Name_i-L : Logarithm on ith feature
        # Name_i-M-Name_j : Multiplication of ith feature with feature jth
        # Name_i-D-Name_j : Division of ith feature with feature jth

        # A small number on the form of a epsilon is being used to avoid NANs because some functions are 0 sensitive,
        # such as the the division by 0. Moreover, a separate list "combinations" is used to keep track the combinations
        # of ith and jth features, and so not to generate duplicate features when multiplying ith feature with jth
        # feature and vice versa (as they are the same number).

        for i in range(len(df.columns)):
            names.append(df.columns[i] + '-I')
            df_combined = pd.concat((df_combined, np.divide(np.ones(df.shape[0]), df.loc[:, df.columns[i]])),
                                    axis=1, ignore_index=True)

            names.append(df.columns[i] + '-L')
            df_combined = pd.concat((df_combined, pd.Series(np.log(np.array(df.loc[:, df.columns[i]]) + 1))),
                                    axis=1, ignore_index=True)

            for j in range(len(df.columns)):
                if i != j:
                    current_combination = str(i) + str(j)
                    if current_combination not in combinations:
                        combinations.append(current_combination)
                        names.append(df.columns[i] + '-M-' + df.columns[j])
                        df_combined = pd.concat((df_combined,
                                                 np.multiply(df.loc[:, df.columns[i]], df.loc[:, df.columns[j]])),
                                                axis=1, ignore_index=True)
                    names.append(df.columns[i] + '-D-' + df.columns[j])
                    df_combined = pd.concat((df_combined,
                                             pd.Series(np.divide(df.loc[:, df.columns[i]],
                                                                 np.array(df.loc[:, df.columns[j]]) + epsilon))),
                                            axis=1, ignore_index=True)

        # The source variables are concatenated with the target variables, infinite numbers are replaced by NANs, and
        # thus any feature with a NAN is removed.
        df_combined.columns = names
        df_combined = df_combined.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

        # The list of names is updated with the new features that do not generated infinite values.
        names = list(df_combined.columns)

        # Information features are included back into the new DataFrame, and thus the columns' names are updated.
        df_combined = pd.concat((df_combined, df_information_combined), axis=1, ignore_index=True)
        df_combined.columns = names + information_features_combined

        return df_combined

    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the DataFrame df, and filters the valid rows back to a
        new df that would then be returned.

        :param pd.DataFrame df: DataFrame with non-normalized, source variables.
        :param string method: Type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame.
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]
        second_value = df.Second

        # The calibration DataFrame is obtained using the "Second" column, as a subset of the calibration phase is taken
        # as a base to further filter samples in the P300 test.
        df_calibration = (df[(df.Second > second_ranges['scaling'][0]) & (df.Second < second_ranges['scaling'][1])]
                          .drop('Second', axis=1))
        df = df.drop('Second', axis=1)

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df_calibration.quantile(q=.25)
            q3 = df_calibration.quantile(q=.75)
            iqr = df_calibration.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]

        # The "Second" column is set back to the original DataFrame, according to the removed rows.
        df['Second'] = second_value[df.index]

        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')

        return df.reset_index(drop=True)

    def normalize(df, method):
        """
        Uses a normalization method to

        :param pd.DataFrame df: Filtered DataFrame according to the "remove_outliers" method used.
        :param string method: Type of normalization method used.
        :return pd.DataFrame: Normalized DataFrame according to the normalization method selected.
        """

        second_value = df.Second

        # A calibration DataFrame is generated, on which parameters would be created to then be applied onto the whole
        # DataFrame, the ranges of second are usually on the Eyes Open (EO) phase when using the P300 test.
        df_calibration = (df[(df.Second > second_ranges['normalize'][0]) & (df.Second < second_ranges['normalize'][1])]
                          .drop('Second', axis=1))
        df = df.drop('Second', axis=1)
        column_names = df.columns

        # A switch case selects a normalization method to be used on the whole DataFrame.
        scaler = StandardScaler().fit(df_calibration) if method == 'standard' else MinMaxScaler().fit(df_calibration)
        df = pd.DataFrame(scaler.transform(df), columns=column_names)

        # The second column is set back to the original, normalized DataFrame.
        df['Second'] = second_value

        # Furthermore, the DataFrame is subset on only containing samples that correspond to outside the calibration
        # phase, as this is the last function applied to the DataFrame, and thus the DataFrame would be fitted into a
        # Machine Learning regression model.
        df = df[df.Second > second_ranges['normalize'][1]]

        print(df)

        return df

    # DataFrame with base features is gathered using the "eeg_analysis" function.
    df_feat = eeg_analysis()
    df_feat['Second'] = pd.to_numeric(df_feat.Second)

    # Information features are defined, these consist on columns for indexing such as "Second", and the target variable.
    information_features = ['Second', 'Subject', 'Repetition', 'FAS']

    print(len(df_feat.Subject.unique()), df_feat.Subject.unique())

    # A separate, empty DataFrame is created, as outliers removal, normalization and feature combination is done for
    # each subject, and thus a for loop needs to be used to iterate over all subjects available.
    df_total = pd.DataFrame()

    # The following for loop iterates over all available subjects, and applies the previously defined functions
    # (outlier removal, normalization, feature combination), the resulted DataFrame is then concatenated into a final
    # DataFrame which contains the processed data for each subject.
    for subject in df_feat.Subject.unique():
        print('Current Subject', subject)

        # The df_filtered is created, which has the data from the current "subject".
        df_filtered = df_feat[df_feat.Subject == subject]

        # A separate DataFrame with information features is kept away, as these features are relevant, but they have
        # nothing to do with normalization methods or outlier removal methods. It is worth noting that the "Second"
        # feature is kept, as it is an important indexer in order to define the limits during the calibration phase.
        df_information = df_filtered[information_features[1:]].reset_index(drop=True)
        df_filtered = df_filtered.drop(information_features[1:], axis=1)

        # The three functions are applied to the filtered DataFrame, the resulting DataFrame is then saved in a separate
        # DataFrame called "df_processed".
        df_processed = normalize(combined_features(remove_outliers(df_filtered, norm_method),
                                                   ['Second']), scaler_method)

        # Information features are manually set to the processed DataFrame, these option was chosen because some of the
        # rows are removed, and so a concatenation between DataFrames would cause an error.
        df_processed['Subject'] = df_information['Subject'][0]
        df_processed['Repetition'] = df_information['Repetition'][0]
        df_processed['FAS'] = df_information['FAS'][0]

        # The processed DataFrame is then concatenated with the empty DataFrame, which would have all subjects' data.
        df_total = pd.concat([df_total, df_processed], ignore_index=True)

    print(len(df_total.Subject.unique()), df_total.Subject.unique())

    return df_total


def create_model(tuned=True, n_features=10):
    """
    The following function created a predictive model using the processed data obtained from the "comb_features"
    function, as outliers are removed, combined features are created, and data is normalized.

    :param bool tuned: Whether the models would be created with tuned parameters or not (n_features and some subjects
    that are causing noise to the model due to outliers that were not entirely detected.
    :param int n_features: Default number of features, this could be changed, but preferably < n_subjects.
    :return None: The created model is saved in a .pkl file, the file name depends on whether Empatica's features were
    included in the model or not.
    """

    # Second and Repetition columns are removed, as only source features, FAS and Subject ID are required.
    x = comb_features().drop(['Second', 'Repetition'], axis=1)

    # If the "tuned" parameters is set to True, then subjects that are making noise to each model would be removed.
    if tuned:
        subjects_to_be_removed = remove_subjects['empatica'] if empatica else remove_subjects['non-empatica']
        x = x.drop(x[x.Subject.isin(subjects_to_be_removed)].index, axis=0)
    x = x.dropna(axis=1).reset_index(drop=True)

    print(len(x.Subject.unique()), x.Subject.unique())
    print(Counter(
        [1 if (x >= 22) else 0 for x in [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print(Counter([1 if (35 > x >= 22) else 0 if x < 22 else 2 for x in
                   [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print([x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()])

    # Evaluacion para MLR Empatica & EEG (9 Feats y [16])
    # 12 0.9 0.8 0.50
    # 11 0.9 0.7 0.58
    # 10 0.9 0.8 0.46
    # 9 0.9 0.8 0.58
    # 8 0.6 0.4 0.09

    # Evaluacion para MLR EEG (10 Feats y [4, 20])
    #   ID       2       3        R
    #           0.79    0.5     -0.38
    # 4         0.69    0.38    -0.05
    # 4 13      0.58    0.25    -0.68
    # 4    20   0.83    0.5     0.072
    #   13      0.62    0.38    -0.027
    #   13 20   0.66    0.42    0.304
    #      20   0.77    0.53    -0.47
    # 4 13 20   0.63    0.27    -0.07

    # 2-Categories
    # 1 9
    # 0 5

    # 3-Categories
    # 2 4
    # 1 5
    # 0 5

    # Fatigue score is popped from the x DataFrame, as these would be the prediction and thus the y.
    y = x.pop('FAS')

    # corr_matrix = x.corr().abs()
    # upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # x = x.drop([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)], axis=1)
    #      Con              Sin
    # 0.64 0.42 -0.074  0.71 0.5 -0.031

    # If the "tuned" parameters is set to True, then the right number of features is selected for each model.
    if tuned:
        n_features = number_features['empatica'] if empatica else number_features['non-empatica']

    # RandomForestRegressor is used as a feature selection method, and thus the importance of each feature is sorted.
    s = pd.Series(index=x.drop(['Subject'], axis=1).columns,
                  data=RandomForestRegressor(random_state=50).fit(x.drop(['Subject'], axis=1),
                                                                  y).feature_importances_).sort_values(ascending=False)
    print(s.head(n_features))

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x = x.loc[:, list(s.index[:n_features]) + ['Subject']]

    # Random Forest
    #  N      2          3       R
    # 20    0.643      0.429    0.06
    # 15    0.643      0.5      0.051
    # 14    0.643      0.5      0.051
    # 13    0.714      0.571    0.092
    # 12    0.714      0.571    0.126
    # 11    0.714      0.571    0.140
    # 10    0.714      0.571    0.17
    # 9     0.714      0.571    0.143
    # 8     0.714      0.571    0.104
    # 7     0.643      0.5      0.09
    # 6     0.643      0.5      0.11
    # 5     0.643      0.429    0.25

    # Linear Regression
    #  N      3          2       R
    # 10    0.786      0.5      -0.383

    # previous_features = set(list(x.drop(['Subject'], axis=1).columns))
    # scores_pearson = [np.abs(pearsonr(y, x[feature])[0]) for feature in x.columns[:N_FEATURES]]
    # p_value = [np.abs(pearsonr(y, x[feature])[1]) for feature in x.columns[:N_FEATURES]]
    # df_correlations = pd.DataFrame({'Feature': x.columns[:N_FEATURES], 'Correlation': scores_pearson, 'P': p_value})
    # df_correlations = (df_correlations[df_correlations.P < 0.05].sort_values('Correlation', ascending=False).round(6)
    #                   .reset_index(drop=True))
    #       Con                 Sin
    # 0.714 0.5 -0.031  0.643 0.429 0.06
    # filtered_features = set(list(df_correlations.Feature))
    # stad_reject_features = previous_features.difference(filtered_features)
    # print(df_correlations.head(df_correlations.shape[0]))
    # x = x.drop(list(previous_features.difference(filtered_features)), axis=1)
    # print('{} Features were rejected'.format(len(stad_reject_features)))
    # print('{} Features were accepted'.format(len(filtered_features)))

    print(x)

    # An empty DataFrame of results is generated, for continuous predictions, as well as 2-class, 3-class categorical.
    df_results = pd.DataFrame(
        columns=['Cont_Pred', 'Cont_True', 'Cat_Pred_2', 'Cat_True_2', 'Cat_Pred_3', 'Cat_True_3'])

    # The following for loop iterates over all subjects, in order to implement a Leave-One-Out (LOO) validation scheme,
    # using approximately 10 subjects, the LOO validation consist on using 90% of the data for training and 10% of the
    # data for testing, as the current subjects' data would be used to test the model, while the rest of the subjects'
    # data would be used for model training.
    for current_subject in x.Subject.unique():

        # Train index are the rows that does not correspond to the current subject's ID, while the test index are the
        # rows that contain the current subject's ID.
        train_index = x[x.Subject != current_subject].index
        test_index = x[x.Subject == current_subject].index

        # Indexes are used to create the set of training and testing data, removing subject as a feature.
        x_train, x_test = x.iloc[train_index, :].drop('Subject', axis=1), x.iloc[test_index, :].drop('Subject', axis=1)
        y_train, y_test = y[train_index], y[test_index]

        # Multiple linear regression is used, and training data is used to train the model.
        model = LinearRegression().fit(x_train, y_train)

        # A raw prediction is done using the training model and the testing rows.
        raw_prediction = model.predict(x_test)

        # The raw prediction is transformed to a final number, because remember that FAS score is assigned to all
        # samples from a subject, and thus we are predicting the FAS score for each row assigned to the subject. So,
        # in order to compare both scores, a single score must be generated, in this case the mean of all predicted
        # scores was used, although, if the mean of score is not in a valid range (0 > x > 50), median is used.
        prediction = np.median(np.round(raw_prediction)) if (0 > mean(raw_prediction) > 50) else round(
            mean(raw_prediction))

        # 2-Class and 3-Class Categorical encoding is applied to the prediction.
        prediction_cat_2 = 1 if (prediction >= 22) else 0
        prediction_cat_3 = 1 if (35 > prediction >= 22) else 0 if prediction < 22 else 2

        # Subject's true FAS score is recovered from the "y_true" pandas series.
        y_true = y_test.head(1).item()

        # 2-Class and 3-Class Categorical encoding is applied to the true value.
        y_true_cat_2 = 1 if (y_true >= 22) else 0
        y_true_cat_3 = 1 if (35 > y_true >= 22) else 0 if y_true < 22 else 2

        # Results are appended into the "df_results" using a dictionary and the zip function.
        df_results = df_results.append(dict(zip(list(df_results.columns),
                                                [prediction, y_true,
                                                 prediction_cat_2, y_true_cat_2,
                                                 prediction_cat_3, y_true_cat_3])), ignore_index=True)
        print(current_subject, prediction, y_true, prediction_cat_2, y_true_cat_2, prediction_cat_3, y_true_cat_3)

    # Each subject's ID is set as a new column in the "df_results" DataFrame.
    df_results['Subject'] = x.Subject.unique()
    print(df_results.head(df_results.shape[0]))

    # The DataFrame's columns are transformed into categorical columns, with their respective order.
    df_results.Cat_True_2 = pd.Categorical(df_results.Cat_True_2, categories=[0, 1], ordered=True)
    df_results.Cat_Pred_2 = pd.Categorical(df_results.Cat_Pred_2, categories=[0, 1], ordered=True)
    df_results.Cat_True_3 = pd.Categorical(df_results.Cat_True_3, categories=[0, 1, 2], ordered=True)
    df_results.Cat_Pred_3 = pd.Categorical(df_results.Cat_Pred_3, categories=[0, 1, 2], ordered=True)

    print(accuracy_score(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(accuracy_score(df_results.Cat_True_3, df_results.Cat_Pred_3))
    print(r2_score(df_results.Cont_True, df_results.Cont_Pred))

    print(classification_report(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(classification_report(df_results.Cat_True_3, df_results.Cat_Pred_3))

    # A simple plot is generated, that related both continuous FAS predictions and true values.
    df_results.loc[:, ['Cont_Pred', 'Cont_True']].sort_values('Cont_True', ascending=True).reset_index(drop=True).plot()
    plt.show()

    # A final Linear Regression model is fitted with all the data available, and thus exported as a .pkl file.
    model = LinearRegression().fit(x.drop('Subject', axis=1), y)
    if empatica:
        dump(model, open('Empatica-Project-ALAS-main/Files/model.pkl', 'wb'))
    else:
        dump(model, open('Empatica-Project-ALAS-main/Files/model_EEG.pkl', 'wb'))


empatica = False
norm_method = 'quantile'  # quantile, z-score
scaler_method = 'minmax'  # minmax, standard
second_ranges = {'normalize': [61, 91],
                 'scaling': [1, 91]}
remove_subjects = {'empatica': [16],
                   'non-empatica': [4, 20]}
number_features = {'empatica': 9,
                   'non-empatica': 10}
create_model(tuned=False)
