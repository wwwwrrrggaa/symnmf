import sys


def euclidean_distance(point1, point2):
    """Return the Euclidean distance between two points."""
    dist_sum = 0
    for i in range(len(point1)):
        dist_sum += (point1[i] - point2[i]) ** 2
    return dist_sum ** 0.5


def k_means_iter(centroids, data):
    """Perform one iteration of k-means and return updated centroids."""
    k = len(centroids)
    clusters = [[] for _ in range(k)]
    for point in data:
        min_dist = float('inf')
        min_index = -1
        for i in range(k):
            dist = euclidean_distance(point, centroids[i])
            if dist < min_dist:
                min_dist = dist
                min_index = i
        clusters[min_index].append(point)

    new_centroids = []
    for cluster in clusters:
        if not cluster:
            new_centroids.append(data[0])
            continue
        new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        new_centroids.append(new_centroid)

    return new_centroids


def k_means(k, iter_count, data):
    """Run k-means clustering and return the final centroids."""
    epsilon = 1e-4
    centroids = data[:k]

    for _ in range(iter_count):
        new_centroids = k_means_iter(centroids, data)
        converged = True
        for i in range(k):
            if euclidean_distance(centroids[i], new_centroids[i]) >= epsilon:
                converged = False
                break
        centroids = new_centroids
        if converged:
            break

    return centroids
