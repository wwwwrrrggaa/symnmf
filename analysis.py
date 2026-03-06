import sys

from sklearn.metrics import silhouette_score

import symnmf
import kmeans


def read_data(file_name):
    """Read data points from file and return as a list of lists."""
    data = []
    with open(file_name, "r") as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.split(",")])
    return data


def main():
    """Parse arguments, run SymNMF and KMeans, and print silhouette scores."""
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    data = read_data(file_name)

    try:
        H = symnmf.symnmf(data, k)
        nmf_labels = [row.index(max(row)) for row in H]
        nmf_score = silhouette_score(data, nmf_labels)
        print(f"nmf: {nmf_score:.4f}")
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        final_centroids = kmeans.k_means(k, 300, data)

        kmeans_labels = []
        for point in data:
            min_dist = float("inf")
            closest_idx = -1
            for i, centroid in enumerate(final_centroids):
                dist = kmeans.euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            kmeans_labels.append(closest_idx)

        kmeans_score = silhouette_score(data, kmeans_labels)
        print(f"kmeans: {kmeans_score:.4f}")
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
