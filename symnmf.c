#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "symnmf.h"

int is_valid_integer(const char* str)
{
    char* endptr;
    const char* p;
    if (str == NULL || *str == '\0') return 0;


    for (p = str; *p != '\0'; p++)
    {
        if (*p == '.') return 0;
    }

    strtol(str, &endptr, 10);
    return (*endptr == '\0');
}

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

double** sym_c(double** data_points, int n, int dim)
{
    double** A = allocate_matrix(n, n);
    int i, j;
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

double** ddg_c(double** A, int n)
{
    double** D = allocate_matrix(n, n);
    double sum;
    int i,j;
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

double** norm_c(double** A, double** D, int n)
{
    double** W = allocate_matrix(n, n);
    int i,j;
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

void matrix_multiply(double** res, double** A, double** B, int rowsA, int colsA, int colsB)
{   int i,j,k;
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

double** symnmf_c(double** H, double** W, int n, int k, int max_iter, double eps)
{
    double** H_prev = allocate_matrix(n, k);
    double** HHH = allocate_matrix(n, k);
    double** HT = allocate_matrix(k, n);
    double** HHT = allocate_matrix(n, n);
    double** WH = allocate_matrix(n, k);
    double f_norm, diff;
    int i,j,iter;

    if (!H_prev || !HHH || !HT || !HHT || !WH)
    {
        free_matrix(H_prev, n);
        free_matrix(HHH, n);
        free_matrix(HT, k);
        free_matrix(HHT, n);
        free_matrix(WH, n);
        return NULL;
    }

    for (iter = 0; iter < max_iter; iter++)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                H_prev[i][j] = H[i][j];
            }
        }

        symnmf_iter(H, H_prev, W, HHH, HT, HHT, WH, n, k);

        f_norm = 0.0;

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                diff = H[i][j] - H_prev[i][j];
                f_norm += diff * diff;
            }
        }


        if (f_norm < eps)
        {
            break;
        }
    }

    free_matrix(H_prev, n);
    free_matrix(HHH, n);
    free_matrix(HT, k);
    free_matrix(HHT, n);
    free_matrix(WH, n);

    return H;
}

int is_valid_goal(const char* str)
{
    return strcmp(str, "sym") == 0 || strcmp(str, "ddg") == 0 ||
           strcmp(str, "norm") == 0 || strcmp(str, "symnmf") == 0;
}
int is_valid_file_name(const char* str)
{   FILE* file;
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

double** read_points(const char* file_name, int* n, int* d)
{
    int rows = 0;
    int cols = 1; /* Assuming at least one column */
    char c;
    int first_line = 1;
    double** data_points;
    int i, j;
    FILE* file;

    /* Removed premature assignment of *n and *d here */

    file = fopen(file_name, "r");
    if (!file)
    {
        return NULL;
    }

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

    /* Assign calculated dimensions to output pointers here */
    *n = rows;
    *d = cols;

    rewind(file);

    data_points = allocate_matrix(rows, cols);

    if (!data_points)
    {
        fclose(file);
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            if (fscanf(file, "%lf", &data_points[i][j]) != 1)
            {
                free_matrix(data_points, i + 1);
                fclose(file);
                return NULL;
            }

            if (j < cols - 1)
            {
                fgetc(file);
            }
        }
    }

    fclose(file);
    return data_points;
}


void print_matrix(double** data_points, int n, int d)
{   int i, j;
    for (i = 0; i < n;i++)
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

double** perform_goal(double** data_points,int n,int dim,char *goal)
{
    double **A, **D, **W;
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
        if (!D) {
            free_matrix(A, n);
            return NULL;
        }
        W = norm_c(A, D, n);
        free_matrix(A, n);
        free_matrix(D, n);
        return W;
    }

}


int main(int argc, char** argv)
{
    char* input_file;
    char* goal;
    int n,d;
    double** data_points,**output;

    if (argc != 3)
    {
        printf("An Error Has Occurred\n");

        return 1;
    }

    goal = argv[1];
    input_file = argv[2];
    if (!is_valid_goal(goal))
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    if (!is_valid_file_name(input_file))
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    data_points = read_points(input_file, &n, &d);

    if (data_points == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    output=perform_goal(data_points, n, d, goal);
    print_matrix(output, n, n);
    free_matrix(data_points, n);
    free_matrix(output, n);




    return 0;
}
