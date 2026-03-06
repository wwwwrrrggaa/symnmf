import sys
import os
import math
import numpy as np
import symnmfmodule

###constant values
np.random.seed(1234)
eps = 1e-4
max_iter = 300


def read_file(file_name):
    ### Reads the data points from the given file and returns a list of lists (matrix).
    data_points = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                row = list(map(float, line.split(",")))
                data_points.append(row)
    return data_points


def print_matrix(matrix):
    ## Prints the given matrix with each element formatted to 4 decimal places, separated by commas.
    for row in matrix:
        print(",".join(f"{x:.4f}" for x in row))


def symnmf(data_points, k):
    W = symnmfmodule.norm(data_points)
    total_sum = sum(sum(row) for row in W)
    m = total_sum / (len(W) * len(W[0]))
    # fmt: off
    H_0 = [[np.random.uniform(0, 2 * math.sqrt(m / k)) for _ in range(k)] for _ in range(len(data_points))]
    # fmt:on
    H = symnmfmodule.symnmf(W, H_0, eps, max_iter)
    return H


def sym(data_points):
    return symnmfmodule.sym(data_points)


def ddg(data_points):
    return symnmfmodule.ddg(data_points)


def norm(data_points):
    return symnmfmodule.norm(data_points)


def main():
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

    if not (os.path.isfile(file_name) and file_name.endswith(".txt")):
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
