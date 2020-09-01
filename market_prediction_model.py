import os
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, roc_curve, f1_score, accuracy_score, recall_score, precision_score


def compute_metrics_standardized_confident(y_predicted, x_test, y_test, confidence):
    y_predicted_confident = []
    y_test_confident = []
    for i in range(len(y_predicted)):
        # Confident if it predicts very low or very high
        if y_predicted[i] > confidence or y_predicted[i] < 1 - confidence:
            y_predicted_confident.append(y_predicted[i])
            y_test_confident.append(y_test[i])

    fpr, tpr, thresholds = roc_curve(y_test_confident, y_predicted_confident)
    au_roc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_predicted_binary = [0 if i < optimal_threshold else 1 for i in y_predicted_confident]

    accuracy = accuracy_score(y_test_confident, y_predicted_binary)
    sensitivity = recall_score(y_test_confident, y_predicted_binary)
    # Lazy to calculate
    specificity = -1
    precision = precision_score(y_test_confident, y_predicted_binary)
    cm = confusion_matrix(y_test_confident, y_predicted_binary)

    return accuracy, sensitivity, specificity, precision, au_roc, cm


'''
This function takes in the directory path and returns a numpy array of all the data 
in all the csv files in that directory
'''
def extract_directory_data_user(dir_path):
    data_X = []
    data_Y = []
    for csv_file in os.listdir(dir_path):
        if csv_file[-3:] != "csv":
            continue
        csv_path = os.path.join(dir_path, csv_file)
        temp_X, temp_Y = extract_csv_data_user(csv_path)
        data_X.append(temp_X)
        data_Y.append(temp_Y)

    data_X = pd.concat(data_X, ignore_index=True, axis=0)
    data_Y = pd.concat(data_Y, ignore_index=True).to_numpy().astype(float)
    data_X, standard_scalar = data_preprocess(data_X)
    data_X, data_Y = shuffle(data_X, data_Y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=0)
    x_train, y_train = balance_data(x_train, y_train)
    print(f"Train data shape: {data_X.shape}")
    return x_train, x_test, y_train, y_test, standard_scalar


'''
This function takes in the csv_path and returns a list of the data in the csv file,
extracting only the relevant information
'''
def extract_csv_data_user(csv_path):
    data = pd.read_csv(csv_path)
    data_X = data.iloc[:, 1:-3]
    data_Y = data.iloc[:, -1]
    return data_X, data_Y


'''
This function takes in a row of data from the csv file and returns the preprocessed numpy array of data
by extracting the relevant information, normalizing and converting the data to numerical.
The first 17 values are user features, the last 10 values are 
'''
def data_preprocess(data, standard_scalar=None):
    data = data.fillna(data.mean())
    data = data.to_numpy().astype(float)

    if standard_scalar is None:
        standard_scalar = preprocessing.StandardScaler()
        data_standardized = standard_scalar.fit_transform(data)
    else:
        data_standardized = standard_scalar.transform(data)

    return data_standardized, standard_scalar


def balance_data(data_X, data_Y):
    data_balanced_X, data_balanced_Y = [], []
    count_0, count_1 = 0, 0
    for label in data_Y:
        if label == [0]:
            count_0 += 1
        else:
            count_1 += 1
    data_size = min(count_0, count_1)
    count_0, count_1 = 0, 0
    for idx, label in enumerate(data_Y):
        if label == [0]:
            count_0 += 1
            if count_0 < data_size:
                data_balanced_X.append(data_X[idx])
                data_balanced_Y.append(label)
        else:
            count_1 += 1
            if count_1 < data_size:
                data_balanced_X.append(data_X[idx])
                data_balanced_Y.append(label)
    print(count_0, count_1)
    return np.asarray(data_balanced_X), np.asarray(data_balanced_Y)


def tree_feature_selection(x_train, x_test, y_train):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train_new = model.transform(x_train)
    x_test_new = model.transform(x_test)
    return x_train_new, x_test_new


def svc_feature_selection(x_train, x_test, y_train):
    lsvc = LinearSVC(C=0.005, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    x_train_new = model.transform(x_train)
    x_test_new = model.transform(x_test)
    print(x_train_new.shape)
    return x_train_new, x_test_new, model


if __name__ == "__main__":
    market_data_directory = "market_data"
    x_train, x_test, y_train, y_test, standard_scalar = extract_directory_data_user(market_data_directory)
    x_train, x_test, fs_model = svc_feature_selection(x_train, x_test, y_train)

    model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
                              keras.layers.Dense(8, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

    y_predicted = model.predict(x_test)

    results = compute_metrics_standardized_confident(y_predicted, x_test, y_test, confidence=0.7)
    print(results[0])
