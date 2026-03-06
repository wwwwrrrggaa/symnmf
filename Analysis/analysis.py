import sys
import symnmf
import kmeans
from sklearn.metrics import silhouette_score


def read_data(file_name):
    data = []
    with open(file_name, "r") as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.split(",")])
    return data


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py k file_name")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except ValueError:
        print("Invalid arguments")
        sys.exit(1)

    data = read_data(file_name)

    # --- SymNMF Analysis ---
    try:
        # Calculate H matrix using the C extension
        H = symnmf.symnmf(data, k)

        # Derive clusters: returns index of the max element in each row of H
        nmf_labels = [row.index(max(row)) for row in H]

        nmf_score = silhouette_score(data, nmf_labels)
        print(f"nmf: {nmf_score:.4f}")
    except Exception as e:
        # Print error details to stderr to avoid confusing the parser
        print(f"SymNMF Error: {e}", file=sys.stderr)

    # --- KMeans Analysis ---
    try:
        # Run KMeans using your imported kmeans.py module
        # k_means(k, iter_count, data) - using 300 iterations as standard
        centroids = kmeans.k_means(k, 300, data)

        # Assign each point to the closest centroid to get labels
        kmeans_labels = []
        for point in data:
            min_dist = float("inf")
            closest_idx = -1

            for i, centroid in enumerate(centroids):
                dist = kmeans.euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            kmeans_labels.append(closest_idx)

        kmeans_score = silhouette_score(data, kmeans_labels)
        print(f"kmeans: {kmeans_score:.4f}")
    except Exception as e:
        print(f"KMeans Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
