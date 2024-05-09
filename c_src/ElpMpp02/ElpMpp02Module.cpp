#include <Python.h>
#include "ElpMpp02.h"

static PyObject* py_get_lunar_coordinates(PyObject* self, PyObject* args) {
    double JD;
    if (!PyArg_ParseTuple(args, "d", &JD)) {
        return NULL; // If parsing fails, return NULL
    }

    Elp_facs facs;
    Elp_paras paras;
    Elp_coefs coefs;
    int corr;
    double X = 0.0, Y = 0.0, Z = 0.0, T;

    // Setup and computations (as from the original C code)
    corr = 0; // LLR data
    setup_parameters(corr, paras, facs);
    setup_Elp_coefs(coefs, facs);
    T = (JD - 2451545.0) / 36525.0;
    getX2000(T, paras, coefs, X, Y, Z);

    // Create a Python tuple to return multiple values
    PyObject* result = Py_BuildValue("(ddd)", X, Y, Z);
    return result;
}



static PyMethodDef ElpMpp02Methods[] = {
    {"get_lunar_coordinates", py_get_lunar_coordinates, METH_VARARGS, "Calculate lunar coordinates given a Julian date."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef ElpMpp02Module = {
    PyModuleDef_HEAD_INIT,
    "lunar",   // name of module
    "Module to calculate lunar coordinates.",  // module documentation
    -1,         // size of per-interpreter state of the module
    ElpMpp02Methods
};

PyMODINIT_FUNC PyInit_ElpMpp02(void) {
    return PyModule_Create(&ElpMpp02Module);
}