#ifndef SYMNMF_SYMNMF_H
#define SYMNMF_SYMNMF_H

#include <stdlib.h>
#include <math.h>

/* Helper functions */
void free_matrix(double** data, int n);
double** allocate_matrix(int rows, int cols);

/* Core algorithm functions */
double** sym_c(double** data_points, int n, int dim);
double** ddg_c(double** A, int n);
double** norm_c(double** A, double** D, int n);
double** symnmf_c(double** H, double** W, int n, int k, int max_iter, double eps);

#endif /* SYMNMF_SYMNMF_H */
