import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gc
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


# seed for numpy and sklearn
random_state = 1
np.random.seed(random_state)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

    print('training set shape = {0}'.format(train.shape))
    print('testing set shape = {0}'.format(test.shape))

    train.fillna(0)
    test.fillna(0)

    y_train = train['death']
    x_train = train.drop(['death'], axis=1)
    y_test = test['death']
    x_test = test.drop(['death'], axis=1)

    print('count class 0 training = {0}'.format(len(y_train[y_train == 0])))
    print('count class 1 training = {0}'.format(len(y_train[y_train == 1])))
    print('count class 0 testing = {0}'.format(len(y_test[y_test == 0])))
    print('count class 1 testing = {0}'.format(len(y_test[y_test == 1])))

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
    model.fit(x_train, y_train, epochs=150, batch_size=5, verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("accuracy: {0:.4}\n".format(scores[1] * 100))

    y_pred = model.predict_proba(x_test)
    y_pred = y_pred.argmax(axis=-1)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    class_names = ['Deaths, Survivals']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


if __name__ == '__main__':
    main()
    gc.collect()
