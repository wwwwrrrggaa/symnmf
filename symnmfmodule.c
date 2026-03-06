#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

/* Unpacks a single Python list row into pre-allocated C array dest of length cols. */
static int unpack_single_row(PyObject* row_obj, double* dest, int cols)
{
    int j;
    PyObject* val_obj;
    double val;
    if (!PyList_Check(row_obj)) return 0;
    for (j = 0; j < cols; j++)
    {
        val_obj = PyList_GetItem(row_obj, j);
        val = PyFloat_AsDouble(val_obj);
        if (PyErr_Occurred()) return 0;
        dest[j] = val;
    }
    return 1;
}

/* Converts a Python list of lists to a double** matrix of given rows x cols. */
static double** unpack_python_matrix(PyObject* py_list, int rows, int cols)
{
    double** matrix;
    int i, ci;
    if (!PyList_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            for (ci = 0; ci < i; ci++) free(matrix[ci]);
            free(matrix);
            PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
            return NULL;
        }
        if (!unpack_single_row(PyList_GetItem(py_list, i), matrix[i], cols)) {
            for (ci = 0; ci <= i; ci++) free(matrix[ci]);
            free(matrix);
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }
    }

    return matrix;
}

/* Converts a double** matrix of rows x cols to a Python list of lists. */
static PyObject* matrix_to_pyobject_list(double** matrix, int rows, int cols)
{
    int i, j;
    PyObject* py_list;
    PyObject* row_list;
    PyObject* py_float;
    py_list = PyList_New(rows);
    if (!py_list) return NULL;

    for (i = 0; i < rows; ++i)
    {
        row_list = PyList_New(cols);
        if (!row_list)
        {
            Py_DECREF(py_list);
            return NULL;
        }

        for (j = 0; j < cols; ++j)
        {
            py_float = PyFloat_FromDouble(matrix[i][j]);
            if (!py_float)
            {
                Py_DECREF(row_list);
                Py_DECREF(py_list);
                return NULL;
            }
            PyList_SET_ITEM(row_list, j, py_float);
        }

        PyList_SET_ITEM(py_list, i, row_list);
    }
    return py_list;
}

/* Extracts row and column counts from a Python list of lists into rows and cols. */
static int get_dims(PyObject* py_list, int* rows, int* cols)
{
    PyObject* first_row;
    if (!PyList_Check(py_list)) return 0;
    *rows = (int)PyList_Size(py_list);
    if (*rows == 0) return 0;

    first_row = PyList_GetItem(py_list, 0);
    if (!PyList_Check(first_row)) return 0;

    *cols = (int)PyList_Size(first_row);
    return 1;
}

/* Python binding: computes and returns the similarity matrix for the given data. */
static PyObject* sym(PyObject *self, PyObject *args)
{
    PyObject* X;
    int n, d;
    double** data_points;
    double** A;
    PyObject* Similarity;
    if (!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    A = sym_c(data_points, n, d);
    free_matrix(data_points, n);
    if (!A) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    Similarity = matrix_to_pyobject_list(A, n, n);
    free_matrix(A, n);

    return Similarity;
}

/* Python binding: computes and returns the diagonal degree matrix for the given data. */
static PyObject* ddg(PyObject *self, PyObject *args)
{
    PyObject* X;
    int n, d;
    double** data_points;
    double** A;
    double** D;
    PyObject* DDG;
    if (!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    A = sym_c(data_points, n, d);
    free_matrix(data_points, n);
    if (!A) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    D = ddg_c(A, n);
    free_matrix(A, n);
    if (!D) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    DDG = matrix_to_pyobject_list(D, n, n);
    free_matrix(D, n);

    return DDG;
}

/* Computes the normalized similarity matrix W from data_points (n x d). Returns NULL on failure. */
static double** compute_norm_from_data(double** data_points, int n, int d)
{
    double** A;
    double** D;
    double** W;

    A = sym_c(data_points, n, d);
    if (!A) return NULL;

    D = ddg_c(A, n);
    if (!D) { free_matrix(A, n); return NULL; }

    W = norm_c(A, D, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

/* Python binding: computes and returns the normalized similarity matrix for the given data. */
static PyObject* norm(PyObject *self, PyObject *args)
{
    PyObject* X;
    int n, d;
    double** data_points;
    double** W;
    PyObject* NORM;
    if (!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    W = compute_norm_from_data(data_points, n, d);
    free_matrix(data_points, n);
    if (!W) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    NORM = matrix_to_pyobject_list(W, n, n);
    free_matrix(W, n);

    return NORM;
}

/* Parses symnmf args and unpacks W and H matrices. Returns 1 on success, 0 on failure. */
static int parse_symnmf_args(PyObject* args, double*** W_out, double*** H_out,
                              int* n, int* k, int* max_iter, double* tol)
{
    PyObject *H_obj, *W_obj;
    int n_w, d_w;

    if (!PyArg_ParseTuple(args, "OOdi", &W_obj, &H_obj, tol, max_iter))
        return 0;

    if (!get_dims(H_obj, n, k) || !get_dims(W_obj, &n_w, &d_w)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return 0;
    }
    if (*n != n_w || *n != d_w) {
        PyErr_SetString(PyExc_ValueError, "An Error Has Occurred");
        return 0;
    }

    *H_out = unpack_python_matrix(H_obj, *n, *k);
    *W_out = unpack_python_matrix(W_obj, *n, *n);
    if (!*H_out || !*W_out) {
        free_matrix(*H_out, *n);
        free_matrix(*W_out, *n);
        return 0;
    }
    return 1;
}

/* Python binding: runs SymNMF optimization and returns the final H matrix. */
static PyObject* symnmf(PyObject *self, PyObject *args)
{
    double tol;
    int max_iter, n, k;
    double** H;
    double** W;
    PyObject* OUTPUT;

    if (!parse_symnmf_args(args, &W, &H, &n, &k, &max_iter, &tol))
        return NULL;

    H = symnmf_c(H, W, n, k, max_iter, tol);
    if (!H) {
        free_matrix(W, n);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    OUTPUT = matrix_to_pyobject_list(H, n, k);
    free_matrix(H, n);
    free_matrix(W, n);

    return OUTPUT;
}

static PyMethodDef SymnmfMethods[] = {
    {"sym", (PyCFunction)sym, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", (PyCFunction)ddg, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", (PyCFunction)norm, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, "Perform full symNMF"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymnmfMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}
