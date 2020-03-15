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


def mean_squared_seperation(clusters):
    k_clusters = clusters.shape[0]
    denom = (k_clusters * (k_clusters - 1)) / 2
    sum = 0.0
    for k in range(k_clusters):
        sum += euclidean_distance_squared(clusters[k], clusters[k:])
    return sum / denom


def update_cluster(incoming_matrix, assignments, clusters):
    cluster_its = np.full(clusters.shape[0], range(clusters.shape[0]))
    cluster_row_its = np.full(clusters.shape[1], range(clusters.shape[1]))
    return_cluster = np.empty_like(clusters)
    for i in cluster_its[0:]:
        matrix_data = incoming_matrix[np.where(assignments == i)]
        for r in cluster_row_its[0:]:
            return_cluster[i][r] = np.sum(matrix_data[:, r]) / matrix_data.shape[0]
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


def assign_inputs(input_size, distances):
    assignments = np.full(input_size, range(input_size))
    for it in assignments[0:]:
        assignments[it] = np.where(distances[..., it] == np.amin(distances[..., it]))[0][0]
    return assignments


# training_data = read_file('optdigits/optdigits.train')
training_data = np.asarray([[0, 1, 0.0], [1, 0, 0.0], [2, 0, 0.0], [4, 0, 0.0]]).astype(float)
# test_data = read_file('optdigits/optdigits.test')
data_row_length = training_data.shape[1] - 1
num_of_training_rows = training_data.shape[0]

cluster_size = 2
# random_cluster_center_indices = np.random.choice(num_of_training_rows, cluster_size, replace=False)
# random_cluster_centers = training_data[random_cluster_center_indices, :data_row_length]

random_cluster_centers = np.asarray([[1, 1], [4, 1]]).astype(float)
# updated = update_cluster(training_data[0:, 0:data_row_length])

distance_list = get_distances(training_data, random_cluster_centers)

assignments = assign_inputs(num_of_training_rows, distance_list)

new_clusters = update_cluster(training_data, assignments, random_cluster_centers)

distance_list2 = get_distances(training_data, new_clusters)

assignments2 = assign_inputs(num_of_training_rows, distance_list2)

# print(mean_squared(distance_list2[0][np.where(assignments2 == 0)])) gets correct

new_clusters2 = update_cluster(training_data, assignments2, new_clusters)

# print(new_clusters2)
