// symnmfmodule.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

// Helper function to convert Python list of lists to double**
static double** unpack_python_matrix(PyObject* py_list, int rows, int cols)
{
    if (!PyList_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

    for (int i = 0; i < rows; i++)
    {
        PyObject* row_obj = PyList_GetItem(py_list, i); // Borrowed reference

        if (!PyList_Check(row_obj))
        {
            free(matrix); // Shallow free, rows not alloc'd yet
            PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
            return NULL;
        }

        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            for(int k=0; k<i; k++) free(matrix[k]);
            free(matrix);
            PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
            return NULL;
        }

        for (int j = 0; j < cols; j++)
        {
            PyObject* val_obj = PyList_GetItem(row_obj, j);
            double val = PyFloat_AsDouble(val_obj);

            if (PyErr_Occurred())
            {
                for(int k=0; k<=i; k++) free(matrix[k]);
                free(matrix);
                return NULL;
            }

            matrix[i][j] = val;
        }
    }

    return matrix;
}
// Helper function to convert double** to Python list of lists
static PyObject* matrix_to_pyobject_list(double** matrix, int rows, int cols)
{
    PyObject* py_list = PyList_New(rows);
    if (!py_list) return NULL;

    for (int i = 0; i < rows; ++i)
    {
        PyObject* row_list = PyList_New(cols);
        if (!row_list)
        {
            Py_DECREF(py_list);
            return NULL;
        }

        for (int j = 0; j < cols; ++j)
        {
            PyObject* py_float = PyFloat_FromDouble(matrix[i][j]);
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
// Helper to get logic dimensions
static int get_dims(PyObject* py_list, int* rows, int* cols) {
    if (!PyList_Check(py_list)) return 0;
    *rows = (int)PyList_Size(py_list);
    if (*rows == 0) return 0;

    PyObject* first_row = PyList_GetItem(py_list, 0);
    if (!PyList_Check(first_row)) return 0;

    *cols = (int)PyList_Size(first_row);
    return 1;
}

static PyObject* sym(PyObject *self, PyObject *args)
{
    PyObject* X;
    if(!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    int n, d;
    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    double** data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    double** A = sym_c(data_points, n, d);
    PyObject* Similarity = matrix_to_pyobject_list(A, n, n);

    free_matrix(data_points, n);
    free_matrix(A, n);

    return Similarity;
}

static PyObject* ddg(PyObject *self, PyObject *args)
{
    PyObject* X;
    if(!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    int n, d;
    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    double** data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    double** A = sym_c(data_points, n, d);
    double** D = ddg_c(A, n);

    PyObject* DDG = matrix_to_pyobject_list(D, n, n);

    free_matrix(data_points, n);
    free_matrix(A, n);
    free_matrix(D, n);

    return DDG;
}

static PyObject* norm(PyObject *self, PyObject *args)
{
    PyObject* X;
    if(!PyArg_ParseTuple(args, "O", &X)) {
        return NULL;
    }

    int n, d;
    if (!get_dims(X, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    double** data_points = unpack_python_matrix(X, n, d);
    if (!data_points) return NULL;

    double** A = sym_c(data_points, n, d);
    double** D = ddg_c(A, n);
    double** W = norm_c(A, D, n);

    if (W == NULL)
    {
        free_matrix(data_points, n);
        free_matrix(A, n);
        free_matrix(D, n);
        Py_RETURN_NONE;
    }
    PyObject* NORM = matrix_to_pyobject_list(W, n, n);

    free_matrix(data_points, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);

    return NORM;
}

static PyObject* symnmf(PyObject *self, PyObject *args)
{
    PyObject *H_obj, *W_obj;
    double tol;
    int max_iter;

    if(!PyArg_ParseTuple(args, "OOdi", &W_obj, &H_obj, &tol, &max_iter)) {
        return NULL;
    }

    int n, k;
    if (!get_dims(H_obj, &n, &k)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    int n_w, d_w;
    if (!get_dims(W_obj, &n_w, &d_w)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    if (n != n_w || n != d_w) {
         PyErr_SetString(PyExc_ValueError, "An Error Has Occurred"); // Dimensions mismatch
         return NULL;
    }

    double** H = unpack_python_matrix(H_obj, n, k);
    double** W = unpack_python_matrix(W_obj, n, n);

    if (!H || !W) {
        free_matrix(H, n);
        free_matrix(W, n);
        return NULL;
    }

    double** H_result = symnmf_c(H, W, n, k, max_iter, tol);

    if (H_result == NULL)
    {
        free_matrix(H, n);
        free_matrix(W, n);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    PyObject* OUTPUT = matrix_to_pyobject_list(H_result, n, k);

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
