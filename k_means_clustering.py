# Joseph Starr
# PSU ID: 943356678
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

# READ ME
# This program should be run the same way assignment 3 was run:
# py k_means_clustering.py trainfilepath testfilepath


# Function for getting data
def read_file(file_name):
    with open(file_name) as opened_file:
        return np.asarray(list(csv.reader(opened_file, delimiter=','))).astype(int)


def read_user_input():
    training_file_user_input = sys.argv[1]
    test_file_user_input = sys.argv[2]
    return training_file_user_input, test_file_user_input


# function to plot digits on 8 x 8 grey scale
def plot_digits(data_matrix, cluster, k):
    data = np.row_stack((data_matrix[0:, 0:data_matrix.shape[1] - 1], cluster))
    fig = plt.figure(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(data.shape[0]):
        plt.imshow(data.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
    plt.show()


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


# assign matrix data to clusters using the matrix data and an array of
# the clusters each row in the matrix is assigned to
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


# Assign each row of 'input_size' matrix to a cluster
# by creating vector of size input_size column length,
# to represent each row and which cluster it is assigned to
def assign_inputs(input_size, distances):
    assignments = np.full(input_size, range(input_size))
    for it in assignments[0:]:
        assignments[it] = np.where(distances[..., it] == np.amin(distances[..., it]))[0][0]
    return assignments


def get_entropy(assignments, data_matrix, k_clusters):
    cluster_its = np.full(k_clusters, range(k_clusters))
    data_matrix_row_size = data_matrix.shape[1]
    entropies = []
    # loop through each cluster and sum the entropies
    for c in cluster_its[0:]:
        # labels of data in cluster c
        matrix_data_labels = data_matrix[np.where(assignments == c)][:, data_matrix_row_size - 1]
        num_of_labels = matrix_data_labels.shape[0]
        diff_labels = np.unique(matrix_data_labels)
        sum = 0.0
        counts = []
        for d in range(diff_labels.shape[0]):
            counts.append(np.count_nonzero(matrix_data_labels == diff_labels[d]))
        counts = np.asarray(counts)
        for count in range(counts.shape[0]):
            sum += (counts[count] / num_of_labels) * np.log2(counts[count] / num_of_labels)
        #     put the entropies in a list to return so they are organized by iteration in main loop
        entropies.append(-sum if sum != 0.0 else 0.0)
    return np.asarray(entropies)


def get_mean_entropy(entropies, assignments, data_matrix, k_clusters):
    data_matrix_row_size = data_matrix.shape[1] - 1
    data_matrix_size = data_matrix.shape[0]
    sum = 0.0
    for k in range(k_clusters):
        matrix_data = data_matrix[np.where(assignments == k)][:, data_matrix_row_size]
        sum += (matrix_data.shape[0] / data_matrix_size) * entropies[k]
    return sum


# Get the class each cluster is assigned to
def get_cluster_labels(incoming_clusters, incoming_assignments, data_matrix):
    cluster_its = incoming_clusters.shape[0]
    data_matrix_row_size = data_matrix.shape[1] - 1
    cluster_labels = []
    for c in range(cluster_its):
        matrix_data = data_matrix[np.where(incoming_assignments == c)][:, data_matrix_row_size]
        cluster_labels.append(np.bincount(matrix_data).argmax())
    return cluster_labels


data_recieved = read_user_input()
training_data = read_file(data_recieved[0])
test_data = read_file(data_recieved[1])
data_row_length = training_data.shape[1] - 1
num_of_training_rows = training_data.shape[0]
num_of_test_row = test_data.shape[0]
confusion_matrix_list = []

cluster_sizes = np.asarray([10, 30])
for cluster_size in cluster_sizes[0:]:
    confusion_matrix = np.full((10, 10), 0.0)
    clusters = []
    cluster_labels = []
    assignment = []
    distances = []
    average_mean_squared_list = []
    mean_squared_seperation_list = []
    mean_entropy_list = []
    for _ in range(5):
        # Randomly generate cluster data from training data
        random_cluster_center_indices = np.random.choice(num_of_training_rows, cluster_size, replace=False)
        random_cluster_centers = training_data[random_cluster_center_indices, :data_row_length]
        index = 0
        # while our clusters are changing do...
        while True:
            distances = get_distances(training_data, random_cluster_centers)

            assignment = assign_inputs(num_of_training_rows, distances)
            clusters_to_update = clusters[len(clusters) - 1] if index > 0 else random_cluster_centers
            clusters.append(update_cluster(training_data, assignment, clusters_to_update))

            if index > 1 and np.array_equal(clusters[index], clusters[index - 1]):
                break
            index += 1

        # get all 5 mean squared, average mean, mean square seperation, and entropies for report
        mean_squared_list = []
        for i in range(distances.shape[0]):
            mean_squared_list.append(mean_squared(distances[i]))
        mean_squared_list = np.asarray(mean_squared_list)
        average_mean_squared_list.append(average_mean_squared(mean_squared_list, cluster_size))
        mean_squared_seperation_list.append(mean_squared_seperation(clusters[len(clusters) - 1]))

        entropies = get_entropy(assignment, training_data, cluster_size)
        mean_entropy_list.append(get_mean_entropy(entropies, assignment, training_data, cluster_size))

    average_mean_squared_list = np.asarray(average_mean_squared_list)
    mean_squared_seperation_list = np.asarray(mean_squared_seperation_list)
    mean_entropy_list = np.asarray(mean_entropy_list)
    index_of_best = np.where(average_mean_squared_list == np.amin(average_mean_squared_list))[0][0]

    print('best of: ')
    print('average mean square error: ', average_mean_squared_list[index_of_best])
    print('mean square seperation: ', average_mean_squared_list[index_of_best])
    print('mean entropy: ', mean_entropy_list[index_of_best])

    # final cluster created from training
    final_clusters = clusters[len(clusters) - 1]
    cluster_labels = get_cluster_labels(final_clusters, assignment, training_data)
    # figure out which rows in the data belong to which cluster
    distances = get_distances(test_data, final_clusters)
    assignments = assign_inputs(test_data.shape[0], distances)

    accuracy = 0.0
    # find acccuracy
    for c in range(cluster_size):
        matrix_data = test_data[np.where(assignments == c)]
        matrix_data_labels = test_data[np.where(assignments == c)][:, data_row_length]
        for num in range(10):
            confusion_matrix[cluster_labels[c]][num] += np.where(matrix_data_labels == num)[0].shape[0]
        plot_digits(matrix_data, final_clusters[c], cluster_size)
        number_of_hits = np.where(matrix_data_labels == cluster_labels[c])
        accuracy += number_of_hits[0].shape[0]

    print(accuracy / num_of_test_row)
    print(confusion_matrix)
