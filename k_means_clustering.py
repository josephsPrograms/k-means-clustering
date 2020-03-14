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


# np.full(785, np.random.uniform(-0.5, 0.501, 785))


# training_data = read_file('optdigits/optdigits.train')
training_data = np.asarray([[0, 1], [1, 0], [2, 0], [4, 0]])
# test_data = read_file('optdigits/optdigits.test')
# data_row_length = training_data.shape[1] - 1

# k_cluster = 10
# random_cluster_centers = np.full((10, data_row_length), np.random.uniform(0, 16, data_row_length))
random_cluster_centers = np.asarray([[1, 1], [4, 1]]).astype(float)

data_its = np.full(training_data.shape[0], range(training_data.shape[0]))
m_its = np.full(random_cluster_centers.shape[0], range(random_cluster_centers.shape[0]))
for m_it in m_its[0:]:
    for it in data_its[0:]:
        print(euclidean_distance_squared(training_data[it], random_cluster_centers[m_it]))

