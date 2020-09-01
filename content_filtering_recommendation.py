import os
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle

'''
This function takes in the directory path and returns a numpy array of all the data 
in all the csv files in that directory
'''
def extract_directory_data_user(dir_path):
    csv_data = []
    for csv_file in os.listdir(dir_path):
        if csv_file[-3:] != "csv":
            continue
        csv_path = os.path.join(dir_path, csv_file)
        csv_data.extend(extract_csv_data_user(csv_path))

    csv_data, min_max_scalar = data_preprocess(csv_data)
    print(f"Shape of training data: {csv_data.shape}")
    return csv_data, min_max_scalar


'''
This function takes in the csv_path and returns a list of the data in the csv file,
extracting only the relevant information
'''
def extract_csv_data_user(csv_path):
    csv_data = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for index, row in enumerate(reader):
            if index == 0:
                continue
            csv_data.append(row)
    return csv_data


'''
This function takes in a row of data from the csv file and returns the preprocessed numpy array of data
by extracting the relevant information, normalizing and converting the data to numerical.
The first 17 values are user features, the last 10 values are 
'''
def data_preprocess(data, min_max_scalar=None):
    tolerance_map = {"0": 0,
                     "Below 10%": 1,
                     "Below 25%": 2,
                     "Above 25%": 3}
    downturn_map = {"Pull out all": 0,
                    "Pull out some": 1,
                    "Observe and not invest more": 2,
                    "Don't pull out and invest more": 3}
    data = np.asarray(data).reshape(-1, 42)
    data_relevant = data[:, 7:]
    for idx in range(len(data_relevant)):
        data_relevant[idx, 1] = 0 if data_relevant[idx, 1] == "Male" else 1
        data_relevant[idx, 10] = tolerance_map[data_relevant[idx, 10]]
        data_relevant[idx, 11] = tolerance_map[data_relevant[idx, 11]]
        data_relevant[idx, 13] = downturn_map[data_relevant[idx, 13]]
        for index, value in enumerate(data_relevant[idx, :17]):
            data_relevant[idx, index] = float(value)
    if min_max_scalar == None:
        min_max_scalar = preprocessing.MinMaxScaler()
        data_features_normalized = min_max_scalar.fit_transform(data_relevant[:, 0:17])
        data_features_normalized[0] = data_features_normalized[0] * 5
    else:
        data_features_normalized = min_max_scalar.transform(data_relevant[:, 0:17])
        data_features_normalized[0] = data_features_normalized[0] * 5
    data_relevant[:, 0:17] = data_features_normalized
    pickle.dump(min_max_scalar, open(os.path.join("trained_models", 'content_scalar.pkl'), 'wb'))
    return data_relevant, min_max_scalar


'''
This function takes in the csv_data, and uses the knn algorithm, 
and returns a dictionary of vector mean values to a set of recommended stocks
'''
def construct_user_knn_graph(csv_data):
    k = 1000
    # Use the 17 features to create 100 clusters
    k_mean_model = KMeans(n_clusters=k)
    k_mean_model.fit(csv_data[:, 0:17])

    # Initialise recommendation list
    recommendation_list = []
    for i in range(k):
        # Each cluster contains a map of the stock and how many times it is bought
        recommendation_list.append({})

    # Populate recommendation list
    for user in csv_data:
        user_features = user[0:17].reshape(1, 17)
        cluster = k_mean_model.predict(user_features)[0]
        # Increment stock count
        for stock in user[-10:]:
            if stock in recommendation_list[cluster]:
                recommendation_list[cluster][stock] += 1
            else:
                recommendation_list[cluster][stock] = 1
    for idx, stock_counter in enumerate(recommendation_list):
        stock_counter_list = []
        for stock in stock_counter:
            stock_counter_list.append((stock_counter[stock], stock))
        stock_counter_list.sort(reverse=True)
        top_stock_list = [i[1] for i in stock_counter_list[0:5]]
        recommendation_list[idx] = top_stock_list
    # Return top 5 most bought stocks
    print("Graph and recommendation list constructed")
    pickle.dump(k_mean_model, open(os.path.join("trained_models", "content_model.pkl"), "wb"))
    pickle.dump(recommendation_list, open(os.path.join("trained_models", 'content_list.pkl'), 'wb'))
    return k_mean_model, recommendation_list


def user_attributes_recommendation(query_features, k_mean_model, recommendation_list):
    cluster = k_mean_model.predict(query_features[:, 0:17])[0]
    return recommendation_list[cluster]


def query_user_features(query_uuid, user_data_file, min_max_scalar):
    # Search for user id and scale the features before returning the data
    for csv_file in os.listdir(user_data_file):
        if csv_file[-3:] != "csv":
            continue
        csv_path = os.path.join(user_data_file, csv_file)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            for row in reader:
                if row[1] == query_uuid:
                    query_features, _ = data_preprocess(row, min_max_scalar)
                    return query_features
    return False


# This function takes in a list of query features and returns the recommendations
# for that user
def content_query_API(query_features):
    min_max_scalar = pickle.load(open(os.path.join("trained_models", 'content_scalar.pkl'), 'rb'))
    k_mean_model = pickle.load(open(os.path.join("trained_models", "content_model.pkl"), 'rb'))
    recommendation_list = pickle.load(open(os.path.join("trained_models", "content_list.pkl"), 'rb'))
    query_features, _ = data_preprocess(query_features, min_max_scalar)
    recommendations = user_attributes_recommendation(query_features, k_mean_model, recommendation_list)
    return recommendations


def build_model():
    # Parameters
    query_uuid = "08c8d13e-9022-4caa-a4c6-edd6383be970"
    user_data_directory = "user_data"

    # Load data ONCE
    csv_data, min_max_scalar = extract_directory_data_user(user_data_directory)
    # Create model ONCE
    k_mean_model, recommendation_list = construct_user_knn_graph(csv_data)

    # Use model for all subsequent predictions
    query_features = query_user_features(query_uuid, user_data_directory, min_max_scalar)
    if query_features is False:
        print("Could not find user")
    else:
        recommendations = user_attributes_recommendation(query_features, k_mean_model, recommendation_list)
        print(f"Recommended stocks for user {query_uuid}: {recommendations}")


# Example usage
# build_model()
# features = ['1', '0b595d3a-f7ad-432f-b9f6-3c9ce2e00e7d', 'Nealy', 'Iley', 'niley1', 'niley1@histats.com', 'G5Odx2', '75', 'Male', '2611484', '1500000', '132278', '135966', '70', '11528', '2', '4', '0', 'Below 10%', '5', "Don't pull out and invest more", '82920', '1895176', '8235270', 'CNXN', 'VNO^L', 'CBZ', 'CHFC', 'Consumer Services', 'Catalog/Specialty Distribution', 'OSMT', 'RHHBY', 'ARCT', 'ITCI', 'BTAI', 'LJPC', 'WFC^J', 'OI', 'LRCX', 'WBA', '1/3/1981', 'SELB']
# print(content_query_API(features))
