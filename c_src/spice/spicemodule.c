#include <Python.h>
#include "SpiceUsr.h"

static PyObject* py_spkezr(PyObject *self, PyObject *args) {
    ConstSpiceChar     *targ;
    SpiceDouble         et;
    ConstSpiceChar     *ref;
    ConstSpiceChar     *abcorr;
    ConstSpiceChar     *obs;
    SpiceDouble         starg[6];
    SpiceDouble         lt;   

    // Parse Python arguments to C types
    if (!PyArg_ParseTuple(args, "sdsds", &targ, &et, &ref, &abcorr, &obs))
        return NULL;

    // Call the CSPICE function
    spkezr_c(targ, et, ref, abcorr, obs, starg, &lt);

    // Build Python tuple from C types
    return Py_BuildValue("(dddddd)d", starg[0], starg[1], starg[2], starg[3], starg[4], starg[5], lt);
}

static PyObject* py_furnsh(PyObject *self, PyObject *args) {
    ConstSpiceChar *file;

    if (!PyArg_ParseTuple(args, "s", &file))
        return NULL;

    furnsh_c(file);

    Py_RETURN_NONE;
}

static PyMethodDef SpiceMethods[] = {
    {"spkezr", py_spkezr, METH_VARARGS,
     "-Brief_I/O \n"
   "VARIABLE  I/O  DESCRIPTION\n"
   "--------  ---  --------------------------------------------------\n"
   "targ       I   Target body.\n"
   "et         I   Observer epoch.\n"
   "ref        I   Reference frame of output state vector.\n"
   "abcorr     I   Aberration correction flag.\n"
   "obs        I   Observing body.\n"
   "starg      O   State of target.\n"
   "lt         O   One way light time between observer and target.\n"},
    {"furnsh", py_furnsh, METH_VARARGS,
     "-Brief_I/O \n"
    "VARIABLE  I/O  DESCRIPTION\n"
    "--------  ---  --------------------------------------------------\n"
    "file       I   Name of SPICE kernel file to load."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef spicemodule = {
    PyModuleDef_HEAD_INIT,
    "spicemodule",
    NULL,  // Module documentation, if any
    -1,   // Module keeps state in global variables
    SpiceMethods
};

PyMODINIT_FUNC PyInit_spice(void) {
    return PyModule_Create(&spicemodule);
}
