import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


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
                dataset[key][dataset[key] == 'Unknown'] = 0
                dataset[key] = pd.to_numeric(dataset[key])

    return dataset


def main():

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

    train.fillna(0)
    test.fillna(0)

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
    :return: NN model
    """
    model = Sequential()
    model.add(Dense(70, input_dim=151, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=150, batch_size=5, verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    main()
