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


def get_entropy(assignments, data_matrix, k_clusters):
    cluster_its = np.full(k_clusters, range(k_clusters))
    data_matrix_row_size = data_matrix.shape[1]
    entropies = []
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
        entropies.append(-sum if sum != 0.0 else 0.0)
    return np.asarray(entropies)


def get_mean_entropy(entropies, assignments, data_matrix, k_clusters):
    data_matrix_row_size = data_matrix.shape[1] - 1
    data_matrix_size = data_matrix.shape[0]
    labels = data_matrix[:, data_matrix_row_size]
    sum = 0.0
    for k in range(k_clusters):
        # print(entropies[k])
        matrix_data = data_matrix[np.where(assignments == k)][:, data_matrix_row_size - 1]
        sum += (matrix_data.shape[0] / data_matrix_size) * entropies[k]
    return sum


training_data = read_file('optdigits/optdigits.train')
# training_data = np.asarray([[0, 1, 1.0], [1, 0, 1.0], [2, 0, 0.0], [4, 0, 1.0]]).astype(float)
# test_data = read_file('optdigits/optdigits.test')
data_row_length = training_data.shape[1] - 1
num_of_training_rows = training_data.shape[0]

cluster_size = 10
# random_cluster_center_indices = np.random.choice(num_of_training_rows, cluster_size, replace=False)
# random_cluster_centers = training_data[random_cluster_center_indices, :data_row_length]

clusters = []

# random_cluster_centers = np.asarray([[1, 1], [4, 1]]).astype(float)
# updated = update_cluster(training_data[0:, 0:data_row_length])

clusters = []
assignments = []
distances = []
average_mean_squared_list = []
mean_squared_seperation_list = []
mean_entropy_list = []
for _ in range(5):
    random_cluster_center_indices = np.random.choice(num_of_training_rows, cluster_size, replace=False)
    random_cluster_centers = training_data[random_cluster_center_indices, :data_row_length]
    index = 0
    while True:
        distances = get_distances(training_data, random_cluster_centers)

        assignment = assign_inputs(num_of_training_rows, distances)

        clusters.append(update_cluster(training_data, assignment, random_cluster_centers))
        if index > 1:
            if np.array_equal(clusters[index], clusters[index - 1]):
                break
        index += 1

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
print(index_of_best)
print('best of: ')
print('average mean square error: ', average_mean_squared_list[index_of_best])
print('mean square seperation: ', average_mean_squared_list[index_of_best])
print('mean entropy: ', mean_entropy_list[index_of_best])
