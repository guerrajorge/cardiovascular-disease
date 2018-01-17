import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gc
from utils.logger import logger_initialization
import argparse
from sklearn.model_selection import StratifiedKFold
import logging
from sklearn.metrics import confusion_matrix


# seed for numpy and sklearn
random_state = 7
np.random.seed(random_state)


def load_data():
    """
    Reading .csv files
    :return: data store in the files with the right pre-processing
    """
    dataset = pd.read_csv('SVR Data/combined_data_m1.csv', index_col='column2')

    # string type columns
    # assoadx and stg2cod are weird variable - need to figure out if it is relevant
    str_cols = ['GENDER', 'STATE', 'GENOTYPE', 'eventyn', 'assoadx', 'stg2cod']
    for key in dataset.keys():

        if is_string_dtype(dataset[key]):
            dataset[key] = dataset[key].str.replace(':.*', '')
            if key not in str_cols:
                # set the unknown values to zero
                dataset.loc[dataset[key] == 'Unknown', key] = 0
                dataset[key] = pd.to_numeric(dataset[key])

    return dataset


def main():

    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)
    logging.getLogger('regular').info('running SVR model')
    logging.getLogger('regular').info('')

    dataset = load_data()

    # assoadx and stg2cod are weird variable - need to figure out if it is relevant
    dataset = dataset.drop(['assoadx', 'stg2cod'], axis=1)

    # encode class values as integers
    encoder = LabelEncoder()
    for key in dataset.keys():
        # masking the NaN values as null
        dataset[key] = dataset[key].factorize()[0]

        if is_string_dtype(dataset[key]):
            dataset[key] = encoder.fit_transform(dataset[key])

    dataset.fillna(0, inplace=True)

    msg = 'dataset shape = {0}'.format(dataset.shape)
    logging.getLogger('regular').info(msg)

    # define 10-fold cross validation test harness
    k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    cv_score = list()

    y = dataset['death'].values
    x = dataset.drop(['death'], axis=1).values

    for k_fold_index, indices in enumerate(k_fold.split(x, y)):

        train_index, test_index = indices

        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(x[train_index])
        x_train = pd.DataFrame(np_scaled)

        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(x[test_index])
        x_test = pd.DataFrame(np_scaled)

        # create model
        model = Sequential()
        model.add(Dense(70, input_dim=151, activation='relu'))
        model.add(Dense(70, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Keras can separate a portion of your training data into a validation dataset and evaluate the performance of
        # your model on that validation dataset each epoch. You can do this by setting the validation_split
        # model.fit(x_train.values, y[train], validation_split=0.33, epochs=150, batch_size=5, verbose=1)
        model.fit(x_train.values, y[train_index], epochs=150, batch_size=5, verbose=1)
        # evaluate the model
        scores = model.evaluate(x_test.values, y[test_index], verbose=1)
        y_pred = model.predict_classes(x_test.values).flatten()

        msg = 'Surviving count data testing = {0}'.format(len(y[test_index][y[test_index] == 0]))
        logging.getLogger('regular').info(msg)
        msg = 'Surviving count data testing = {0}'.format(len(y[test_index][y[test_index] == 1]))
        logging.getLogger('regular').info(msg)
        logging.getLogger('regular').info('')

        # Compute and show confusion matrix
        true_positive, false_positive, false_negative, true_negative = confusion_matrix(y_true=y[test_index],
                                                                                        y_pred=y_pred).flatten()

        msg = 'k-fold {0} model accuracy: {1:.4}'.format(k_fold_index, scores[1] * 100)
        logging.getLogger('regular').info(msg)
        logging.getLogger('regular').info('')

        logging.getLogger('regular').info('confusion matrix\n')
        row = '{0: <10} {1: <10} {2}'.format('', 'Survival', 'Death')
        logging.getLogger('regular').info(row)
        row = '{0: <10} {1: <10} {2}'.format('Survival', true_positive, false_positive)
        logging.getLogger('regular').info(row)
        row = '{0: <10} {1: <10} {2}\n'.format('Death', false_negative, true_negative)
        logging.getLogger('regular').info(row)

        cv_score.append(scores[1] * 100)

    msg = 'Model performance:'
    logging.getLogger('regular').info(msg)
    msg = "average = {0:.4} standard deviation=+/- {1:.4}".format(np.mean(cv_score), np.std(cv_score))
    logging.getLogger('regular').info(msg)


if __name__ == '__main__':
    main()
    gc.collect()
