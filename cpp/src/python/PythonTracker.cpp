#include <Python.h> // to be placed before any other standard library to avoid conflicts
#ifdef _WIN32
#include <windows.h>
#endif

#include "PythonTracker.h"


void PythonTracker::initializeTracker()
{
#ifdef _WIN32
    // Windows requires to specify Python Home into DDL directories in order to resolve:
    // ImportError: DLL load failed while importing _psutil_windows
    std::wstring pythonHome = PYTHON_HOME;
    SetDllDirectoryW(pythonHome.c_str());
#endif

    if (!Py_IsInitialized())
        Py_Initialize();

#ifdef _WIN32
    PyRun_SimpleString("import win_patch_codecarbon");
#endif
#ifdef __linux__
    // Linux requires to add cwd (where Py script are located) into Python sys.path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
#endif

    PyRun_SimpleString("from tracker_control import Tracker;");
}

void PythonTracker::finalizeTracker()
{
    if (Py_IsInitialized())
        Py_Finalize();
}

void PythonTracker::startTracker(const std::string& outputDir, const std::string& outputFile)
{
    const std::string command = "Tracker.start_tracker('" + outputDir + "', '" + outputFile + "')";
    PyRun_SimpleString(command.c_str());
}

void PythonTracker::stopTracker()
{
    PyRun_SimpleString("Tracker.stop_tracker()");
}