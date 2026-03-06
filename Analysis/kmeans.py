import sys

def euclidean_distance(point1, point2):
    dist_sum = 0
    for i in range(len(point1)):
        dist_sum += (point1[i] - point2[i]) ** 2
    return dist_sum ** 0.5

def k_means_iter(centroids, data):
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
    epsilon = 0.001
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

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

    iter_count = 200
    if len(sys.argv) == 3:
        try:
            iter_count = int(sys.argv[2])
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)

    if not (1 < iter_count < 1000):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    data = []
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            point = [float(x) for x in line.split(',')]
            data.append(point)
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    if len(data) == 0:
        print("An Error Has Occurred")
        sys.exit(1)

    d = len(data[0])
    for point in data:
        if len(point) != d:
            print("An Error Has Occurred")
            sys.exit(1)

    if not (1 < k < len(data)):
        print("Incorrect number of clusters!")
        sys.exit(1)

    final_centroids = k_means(k, iter_count, data)

    for centroid in final_centroids:
        formatted_coords = [f"{coord:.4f}" for coord in centroid]
        print(",".join(formatted_coords))

if __name__ == "__main__":
    main()