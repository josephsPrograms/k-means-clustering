# Joseph Starr
# PSU ID: 943356678
import numpy as np
import csv


# Function for getting data
def read_file(file_name):
    with open(file_name) as opened_file:
        return np.asarray(list(csv.reader(opened_file, delimiter=','))).astype(int)


def euclidean_distance_squared(x_vector, m_vector):
    diffs = (x_vector[0:] - m_vector[0:])**2
    sum_diffs = np.sum(diffs)
    return sum_diffs


def mean_squared(incoming_vector):
    sum = np.sum(incoming_vector)
    return sum / incoming_vector.size


def average_mean_squared(mean_squared_vector, k_values):
    sum = np.sum(mean_squared_vector)
    return sum / k_values


def update_cluster(incoming_matrix):
    size_to_label = incoming_matrix.shape[1]
    cluster_its = np.full(size_to_label, range(size_to_label))
    return_cluster = np.full(size_to_label, 0.0)
    for i in cluster_its[0:]:
        return_cluster[i] = np.sum(incoming_matrix[:, i]) / incoming_matrix.shape[0]
    return return_cluster


def get_distances(incoming_data, cluster_centers):
    data_its = np.full(incoming_data.shape[0], range(incoming_data.shape[0]))
    m_its = np.full(cluster_centers.shape[0], range(cluster_centers.shape[0]))
    distances_to_return = []
    for m_it in m_its[0:]:
        sub_distance_list = []
        for it in data_its[0:]:
            sub_distance_list.append(
                euclidean_distance_squared(incoming_data[it, 0:data_row_length], cluster_centers[m_it]))
        distances_to_return.append(sub_distance_list)
    return np.asarray(distances_to_return)


# training_data = read_file('optdigits/optdigits.train')
training_data = np.asarray([[0, 1, 0.0], [1, 0, 0.0], [2, 0, 0.0], [4, 0, 0.0]]).astype(float)
# test_data = read_file('optdigits/optdigits.test')
data_row_length = training_data.shape[1] - 1
num_of_training_rows = training_data.shape[0]

# k_cluster = 10
# random_cluster_center_indices = np.random.choice(num_of_training_rows, k_cluster, replace=False)
# random_cluster_centers = training_data[random_cluster_center_indices, :data_row_length]

random_cluster_centers = np.asarray([[1, 1], [4, 1]]).astype(float)
# updated = update_cluster(training_data[0:, 0:data_row_length])

distance_list = get_distances(training_data, random_cluster_centers)

print(distance_list)
