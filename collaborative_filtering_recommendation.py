import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pickle

def construct_table(user_data, company_data):
    table = np.zeros((user_data.shape[0], company_data.shape[0]))
    for user_idx, row in enumerate(user_data):
        for comp_inx, comp in enumerate(company_data):
            if comp in row:
                table[user_idx, comp_inx] = 1
    return table


def build_user_knn_graph(table, csv_data):
    k = 100
    k_mean_model = KMeans(n_clusters=k)
    k_mean_model.fit(table)
    recommendation_list = []
    for i in range(k):
        # Each cluster contains a map of the stock and how many times it is bought
        recommendation_list.append({})

    # Populate recommendation list
    for idx, user in enumerate(table):
        cluster = k_mean_model.predict(user.reshape(1, -1))[0]
        # Increment stock count
        for stock in csv_data[idx][-10:]:
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
    pickle.dump(k_mean_model, open(os.path.join("trained_models", "collaborative_model.pkl"), "wb"))
    pickle.dump(recommendation_list, open(os.path.join("trained_models", 'collaborative_list.pkl'), 'wb'))
    return k_mean_model, recommendation_list


def taste_breaker_recommendation(query_features, k_mean_model, recommendation_list):
    max_dist = float('-inf')
    max_dist_cluster = 0
    for idx, center in enumerate(k_mean_model.cluster_centers_):
        temp_dist = np.sum(euclidean_distances(query_features, center.reshape(1, -11)))
        if  temp_dist> max_dist:
            max_dist = temp_dist
            max_dist_cluster = idx
    return recommendation_list[max_dist_cluster]



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

    csv_data = data_preprocess(csv_data)
    print(f"Shape of user data: {csv_data.shape}")
    return csv_data


'''
This function takes in a row of data from the csv file and returns the preprocessed numpy array of data
by extracting the relevant information, which is the investment history
'''
def data_preprocess(data):
    data = np.asarray(data).reshape(-1, 42)
    data_relevant = np.delete(data[:, 24:], 16, 1)
    return data_relevant


'''
This function takes in the csv_path and returns a list of all the data in the csv file
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


def extract_csv_data_company(csv_path):
    csv_data = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for index, row in enumerate(reader):
            if index == 0:
                continue
            csv_data.append(row)
    return csv_data


def extract_directory_data_company(dir_path):
    csv_data = []
    for csv_file in os.listdir(dir_path):
        if csv_file[-3:] != "csv":
            continue
        csv_path = os.path.join(dir_path, csv_file)
        csv_data.extend(extract_csv_data_user(csv_path))
    csv_data = np.asarray(csv_data).flatten()
    print(f"Shape of company data: {csv_data.shape}")
    return csv_data


def query_user_features(query_uuid, user_data_file, table):
    # Search for user id and scale the features before returning the data
    for csv_file in os.listdir(user_data_file):
        if csv_file[-3:] != "csv":
            continue
        csv_path = os.path.join(user_data_file, csv_file)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            for idx, row in enumerate(reader):
                if row[1] == query_uuid:
                    return table[idx].reshape(1, -1)
    return False


def user_preference_recommendation(query_features, k_mean_model, recommendation_list):
    cluster = k_mean_model.predict(query_features)[0]
    return recommendation_list[cluster]


def build_model():
    # Parameters
    query_uuid = "08c8d13e-9022-4caa-a4c6-edd6383be970"
    user_data_directory = "user_data"
    company_data_directory = "company_data"
    csv_data = extract_directory_data_user(user_data_directory)
    company_data = extract_directory_data_company(company_data_directory)
    table = construct_table(csv_data, company_data)
    k_mean_model, recommendation_list = build_user_knn_graph(table, csv_data)

    query_features = query_user_features(query_uuid, user_data_directory, table)
    if query_features is False:
        print("Could not find user")
    else:
        recommendations = user_preference_recommendation(query_features, k_mean_model, recommendation_list)
        print(f"Recommended stocks for user {query_uuid}: {recommendations}")
        recommendations = taste_breaker_recommendation(query_features, k_mean_model, recommendation_list)
        print(f"Taste breaker for user {query_uuid}: {recommendations}")


def collaborative_query_API(query_features):
    k_mean_model = pickle.load(open(os.path.join("trained_models", "collaborative_model.pkl"), 'rb'))
    recommendation_list = pickle.load(open(os.path.join("trained_models", "collaborative_list.pkl"), 'rb'))
    company_data = extract_directory_data_company("company_data")
    features = np.zeros(company_data.shape[0])
    for comp_inx, comp in enumerate(company_data):
        if comp in query_features:
            features[comp_inx] = 1
    features = features.reshape(1, -1)
    return user_preference_recommendation(features, k_mean_model, recommendation_list), taste_breaker_recommendation(features, k_mean_model, recommendation_list)

# Example usage
# build_model()
# features = ['1', '0b595d3a-f7ad-432f-b9f6-3c9ce2e00e7d', 'Nealy', 'Iley', 'niley1', 'niley1@histats.com', 'G5Odx2', '75', 'Male', '2611484', '1500000', '132278', '135966', '70', '11528', '2', '4', '0', 'Below 10%', '5', "Don't pull out and invest more", '82920', '1895176', '8235270', 'CNXN', 'VNO^L', 'CBZ', 'CHFC', 'Consumer Services', 'Catalog/Specialty Distribution', 'OSMT', 'RHHBY', 'ARCT', 'ITCI', 'BTAI', 'LJPC', 'WFC^J', 'OI', 'LRCX', 'WBA', '1/3/1981', 'SELB']
# print(collaborative_query_API(features))
