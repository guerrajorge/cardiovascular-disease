import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gc
from sklearn.metrics import confusion_matrix
from utils.logger import logger_initialization
import argparse
import logging

# seed for numpy and sklearn
random_state = 1
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

    train, test = train_test_split(dataset, test_size=0.33, random_state=42)

    msg = 'training set shape = {0}'.format(train.shape)
    logging.getLogger('regular').info(msg)
    msg = 'testing set shape = {0}'.format(test.shape)
    logging.getLogger('regular').info(msg)
    logging.getLogger('regular').info('')

    y_train = train['death']
    x_train = train.drop(['death'], axis=1)
    y_test = test['death']
    x_test = test.drop(['death'], axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(x_train)
    x_train = pd.DataFrame(np_scaled)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(x_test)
    x_test = pd.DataFrame(np_scaled)

    """
    Neural Network model
    """
    model = Sequential()
    model.add(Dense(70, input_dim=151, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Keras can separate a portion of your training data into a validation dataset and evaluate the performance of your
    # model on that validation dataset each epoch. You can do this by setting the validation_split
    model.fit(x_train.values, y_train.values, validation_split=0.33, epochs=150, batch_size=5, verbose=1)
    scores = model.evaluate(x_test.values, y_test.values, verbose=0)

    y_pred = model.predict_classes(x_test.values).flatten()

    msg = 'Surviving count data testing = {0}'.format(len(y_test[y_test == 0]))
    logging.getLogger('regular').info(msg)
    msg = 'Surviving count data testing = {0}'.format(len(y_test[y_test == 1]))
    logging.getLogger('regular').info(msg)
    logging.getLogger('regular').info('')

    # Compute and show confusion matrix
    true_positive, false_positive, false_negative, true_negative = confusion_matrix(y_true=y_test.values,
                                                                                    y_pred=y_pred).flatten()

    msg = 'model accuracy: {0:.4}'.format(scores[1] * 100)
    logging.getLogger('regular').info(msg)
    logging.getLogger('regular').info('')

    logging.getLogger('regular').info('confusion matrix\n')
    row = '{0: <10} {1: <10} {2}'.format('', 'Survival', 'Death')
    logging.getLogger('regular').info(row)
    row = '{0: <10} {1: <10} {2}'.format('Survival', true_positive, false_positive)
    logging.getLogger('regular').info(row)
    row = '{0: <10} {1: <10} {2}\n'.format('Death', false_negative, true_negative)
    logging.getLogger('regular').info(row)


if __name__ == '__main__':
    main()
    gc.collect()
