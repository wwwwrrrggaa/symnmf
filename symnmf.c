#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "symnmf.h"


/* Frees all rows and the outer pointer of a matrix of n rows. */
void free_matrix(double** data, int n)
{
    int i;
    if (data)
    {
        for (i = 0; i < n; i++)
        {
            if (data[i])
            {
                free(data[i]);
            }
        }
        free(data);
    }
}

/* Allocates and returns a rows x cols matrix of doubles, or NULL on failure. */
double** allocate_matrix(int rows, int cols)
{
    double** matrix;
    int i;
    matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix)
    {
        return NULL;
    }
    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!matrix[i])
        {
            free_matrix(matrix, i);
            return NULL;
        }
    }
    return matrix;
}

/* Returns the squared Euclidean distance between two dim-dimensional points. */
double squared_euclidean_distance(double* point1, double* point2, int dim)
{
    int i;
    double sum = 0.0;
    double diff;
    for (i = 0; i < dim; i++)
    {
        diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

/* Computes and returns the n x n similarity matrix A from data_points. */
double** sym_c(double** data_points, int n, int dim)
{
    int i, j;
    double** A = allocate_matrix(n, n);
    if (!A) return NULL;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i][j] = 0;
            }
            else
            {
                A[i][j] = exp(squared_euclidean_distance(data_points[i], data_points[j], dim) / -2.0);
            }
        }
    }
    return A;
}

/* Computes and returns the n x n diagonal degree matrix D from similarity matrix A. */
double** ddg_c(double** A, int n)
{
    double sum;
    int i, j;
    double** D = allocate_matrix(n, n);
    if (!D) return NULL;
    for (i = 0; i < n; i++)
    {
        sum = 0.0;
        for (j = 0; j < n; j++)
        {
            sum += A[i][j];
        }
        for (j = 0; j < n; j++)
        {
            D[i][j] = 0.0;
        }
        D[i][i] = sum;
    }
    return D;
}

/* Computes and returns the normalized similarity matrix W from A and D. */
double** norm_c(double** A, double** D, int n)
{
    int i, j;
    double** W = allocate_matrix(n, n);
    if (!W) return NULL;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]);
        }
    }
    return W;
}

/* Stores the transpose of matrix (n x d) into transposed (d x n). */
void transpose(double** transposed, double** matrix, int n, int d)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            transposed[j][i] = matrix[i][j];
        }
    }
}

/* Multiplies matrix A (rowsA x colsA) by B (colsA x colsB) and stores result in res. */
void matrix_multiply(double** res, double** A, double** B, int rowsA, int colsA, int colsB)
{
    int i, j, k;
    for (i = 0; i < rowsA; i++)
    {
        for (j = 0; j < colsB; j++)
        {
            res[i][j] = 0.0;
            for (k = 0; k < colsA; k++)
            {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Performs one SymNMF update step, writing the new H into H from H_prev and W. */
void symnmf_iter(double** H, double** H_prev, double** W, double** HHH, double** HT, double** HHT, double** WH,
                 int n, int k)
{
    double beta = 0.5;
    double epsilon2 = 1e-6;
    double denominator;
    int i, j;

    matrix_multiply(WH, W, H_prev, n, n, k);

    transpose(HT, H_prev, n, k);
    matrix_multiply(HHT, H_prev, HT, n, k, n);

    matrix_multiply(HHH, HHT, H_prev, n, n, k);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            denominator = HHH[i][j] != 0 ? HHH[i][j] : HHH[i][j] + epsilon2;
            H[i][j] = H_prev[i][j] * (1 - beta + beta * (WH[i][j] / denominator));
        }
    }
}

/* Copies all entries from src into dst (both n x k). */
void copy_matrix(double** dst, double** src, int n, int k)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            dst[i][j] = src[i][j];
        }
    }
}

/* Returns the squared Frobenius norm of the difference (A - B) for n x k matrices. */
double frobenius_norm_diff(double** A, double** B, int n, int k)
{
    double sum = 0.0, diff;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* Frees the five temporary matrices used in symnmf_c. */
void free_temp_matrices(double** H_prev, double** HHH, double** HT,
                        double** HHT, double** WH, int n, int k)
{
    free_matrix(H_prev, n);
    free_matrix(HHH, n);
    free_matrix(HT, k);
    free_matrix(HHT, n);
    free_matrix(WH, n);
}

/* Runs SymNMF optimization on H using W for up to max_iter iterations or until convergence within eps. */
double** symnmf_c(double** H, double** W, int n, int k, int max_iter, double eps)
{
    int iter;
    double** H_prev = allocate_matrix(n, k);
    double** HHH = allocate_matrix(n, k);
    double** HT = allocate_matrix(k, n);
    double** HHT = allocate_matrix(n, n);
    double** WH = allocate_matrix(n, k);

    if (!H_prev || !HHH || !HT || !HHT || !WH)
    {
        free_temp_matrices(H_prev, HHH, HT, HHT, WH, n, k);
        return NULL;
    }

    for (iter = 0; iter < max_iter; iter++)
    {
        copy_matrix(H_prev, H, n, k);
        symnmf_iter(H, H_prev, W, HHH, HT, HHT, WH, n, k);
        if (frobenius_norm_diff(H, H_prev, n, k) < eps)
            break;
    }

    free_temp_matrices(H_prev, HHH, HT, HHT, WH, n, k);
    return H;
}

/* Returns 1 if str is one of the valid goal strings, 0 otherwise. */
int is_valid_goal(const char* str)
{
    return strcmp(str, "sym") == 0 || strcmp(str, "ddg") == 0 ||
           strcmp(str, "norm") == 0;
}

/* Returns 1 if str is an existing .txt file path, 0 otherwise. */
int is_valid_file_name(const char* str)
{
    FILE* file;
    if (strlen(str) < 4 || strcmp(str + strlen(str) - 4, ".txt") != 0)
    {
        return 0;
    }
    file = fopen(str, "r");
    if (file)
    {
        fclose(file);
        return 1;
    }
    return 0;
}

/* Counts the number of rows and columns in file_name, storing in n and d. */
int count_file_dims(const char* file_name, int* n, int* d)
{
    int rows = 0, cols = 1, first_line = 1;
    int c;
    FILE* file = fopen(file_name, "r");
    if (!file) return 0;

    while ((c = fgetc(file)) != EOF)
    {
        if (c == '\n')
        {
            rows++;
            first_line = 0;
        }
        else if (first_line && c == ',')
        {
            cols++;
        }
    }
    fclose(file);
    *n = rows;
    *d = cols;
    return 1;
}

/* Reads data points from file_name into a matrix, setting n and d for dimensions. */
double** read_points(const char* file_name, int* n, int* d)
{
    double** data_points;
    int i, j;
    FILE* file;

    if (!count_file_dims(file_name, n, d)) return NULL;

    file = fopen(file_name, "r");
    if (!file) return NULL;

    data_points = allocate_matrix(*n, *d);
    if (!data_points) { fclose(file); return NULL; }

    for (i = 0; i < *n; i++)
    {
        for (j = 0; j < *d; j++)
        {
            if (fscanf(file, "%lf", &data_points[i][j]) != 1)
            {
                free_matrix(data_points, i + 1);
                fclose(file);
                return NULL;
            }
            if (j < *d - 1) fgetc(file);
        }
    }

    fclose(file);
    return data_points;
}

/* Prints an n x d matrix with values formatted to 4 decimal places. */
void print_matrix(double** data_points, int n, int d)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.4f", data_points[i][j]);
            if (j < d - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Computes and returns the result matrix for the given goal using data_points. */
double** perform_goal(double** data_points, int n, int dim, char* goal)
{
    double** A;
    double** D;
    double** W;
    if (strcmp(goal, "sym") == 0)
    {
        return sym_c(data_points, n, dim);
    }
    else if (strcmp(goal, "ddg") == 0)
    {
        A = sym_c(data_points, n, dim);
        if (!A) return NULL;
        D = ddg_c(A, n);
        free_matrix(A, n);
        return D;
    }
    else
    {
        A = sym_c(data_points, n, dim);
        if (!A) return NULL;
        D = ddg_c(A, n);
        if (!D)
        {
            free_matrix(A, n);
            return NULL;
        }
        W = norm_c(A, D, n);
        free_matrix(A, n);
        free_matrix(D, n);
        return W;
    }
}

/* Validates command-line arguments: checks argc, goal, and file name. */
int validate_args(int argc, char** argv)
{
    if (argc != 3) return 0;
    if (!is_valid_goal(argv[1])) return 0;
    if (!is_valid_file_name(argv[2])) return 0;
    return 1;
}

/* Entry point: validates arguments, reads data, performs goal, and prints the result. */
int main(int argc, char** argv)
{
    char* goal;
    char* input_file;
    int n, d;
    double** data_points;
    double** output;

    if (!validate_args(argc, argv))
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    goal = argv[1];
    input_file = argv[2];
    data_points = read_points(input_file, &n, &d);

    if (data_points == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    output = perform_goal(data_points, n, d, goal);
    if (!output)
    {
        free_matrix(data_points, n);
        printf("An Error Has Occurred\n");
        return 1;
    }

    print_matrix(output, n, n);
    free_matrix(data_points, n);
    free_matrix(output, n);



    return 0;
}
