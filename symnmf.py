import sys
import numpy as np
import symnmfmodule

np.random.seed(1234)
eps = 1e-4
max_iter = 300


def read_file(file_name):
    """Read data points from a comma-separated file and return as a list of lists."""
    data_points = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                row = list(map(float, line.split(",")))
                data_points.append(row)
    return data_points


def print_matrix(matrix):
    """Print each row of the matrix with values formatted to 4 decimal places."""
    for row in matrix:
        print(",".join(f"{x:.4f}" for x in row))


def symnmf(data_points, k):
    """Compute the SymNMF factorization matrix H for the given data and k clusters."""
    W = symnmfmodule.norm(data_points)
    total_sum = sum(sum(row) for row in W)
    m = total_sum / (len(W) * len(W[0]))
    H_0 = [[np.random.uniform(0, 2 * np.sqrt(m / k)) for _ in range(k)] for _ in range(len(data_points))]
    H = symnmfmodule.symnmf(W, H_0, eps, max_iter)
    return H


def sym(data_points):
    """Return the similarity matrix for the given data points."""
    return symnmfmodule.sym(data_points)


def ddg(data_points):
    """Return the diagonal degree matrix for the given data points."""
    return symnmfmodule.ddg(data_points)


def norm(data_points):
    """Return the normalized similarity matrix for the given data points."""
    return symnmfmodule.norm(data_points)


def main():
    """Parse arguments, compute the requested goal, and print the result matrix."""
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    goal = sys.argv[2]
    file_name = sys.argv[3]

    if goal not in ["symnmf", "sym", "ddg", "norm"]:
        print("An Error Has Occurred")
        sys.exit(1)

    data_points = read_file(file_name)
    n = len(data_points)

    if k >= n:
        print("An Error Has Occurred")
        sys.exit(1)

    result = None
    if goal == "symnmf":
        result = symnmf(data_points, k)
    elif goal == "sym":
        result = sym(data_points)
    elif goal == "ddg":
        result = ddg(data_points)
    elif goal == "norm":
        result = norm(data_points)

    if result is not None:
        print_matrix(result)


if __name__ == "__main__":
    main()
