#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "elp2000-82b.h"

// Wrapper for geocentric_moon_position function
static PyObject* py_geocentric_moon_position(PyObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) {
        return NULL;
    }

    spherical_point sp = geocentric_moon_position(t);

    return Py_BuildValue("ddd", sp.longitude, sp.latitude, sp.distance);
}

static PyObject* py_geocentric_moon_position_of_date(PyObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) {
        return NULL;
    }

    spherical_point sp = geocentric_moon_position_of_date(t);

    return Py_BuildValue("ddd", sp.longitude, sp.latitude, sp.distance);
}

static PyObject* py_geocentric_moon_position_cartesian(PyObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) {
        return NULL;
    }

    cartesian_3d_point cp = geocentric_moon_position_cartesian(t);

    return Py_BuildValue("ddd", cp.x, cp.y, cp.z);
}

static PyObject* py_geocentric_moon_position_of_J2000(PyObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) {
        return NULL;
    }

    cartesian_3d_point cp = geocentric_moon_position_cartesian_of_J2000(t);

    return Py_BuildValue("ddd", cp.x, cp.y, cp.z);
}

static PyObject* py_geocentric_moon_position_of_FK5(PyObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) {
        return NULL;
    }

    cartesian_3d_point cp = geocentric_moon_position_cartesian_of_FK5(t);

    return Py_BuildValue("ddd", cp.x, cp.y, cp.z);
}

// List of functions to expose
static PyMethodDef Elp2000Methods[] = {
    {"geocentric_moon_position", py_geocentric_moon_position, METH_VARARGS, "Calculate geocentric moon position"},
    {"geocentric_moon_position_of_date", py_geocentric_moon_position_of_date, METH_VARARGS, "Calculate geocentric moon position of date"},
    {"geocentric_moon_position_cartesian", py_geocentric_moon_position_cartesian, METH_VARARGS, "Calculate geocentric moon position in cartesian coordinates"},
    {"geocentric_moon_position_of_J2000", py_geocentric_moon_position_of_J2000, METH_VARARGS, "Calculate geocentric moon position in cartesian coordinates of J2000"},
    {"geocentric_moon_position_of_FK5", py_geocentric_moon_position_of_FK5, METH_VARARGS, "Calculate geocentric moon position in cartesian coordinates of FK5"},

    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef elp2000module = {
    PyModuleDef_HEAD_INIT,
    "elp2000",  // name of module
    NULL,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    Elp2000Methods
};

// Initialization function for the module
PyMODINIT_FUNC PyInit_elp2000(void) {
    return PyModule_Create(&elp2000module);
}