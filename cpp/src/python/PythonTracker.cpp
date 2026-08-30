#include <Python.h> // must precede other headers to avoid conflicts
#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "PythonTracker.h"


namespace {

    void run(const std::string& code)
    {
        if (PyRun_SimpleString(code.c_str()) != 0) {
            PyErr_Print();
            throw std::runtime_error("measurement bridge failed on: " + code);
        }
    }

    uint64_t envAsU64(const char* name, uint64_t fallback)
    {
        if (const char* v = std::getenv(name)) {
            try {
                return std::stoull(v);
            } catch (const std::exception&) {
                // fall through to the default
            }
        }
        return fallback;
    }

}

void PythonTracker::initializeTracker()
{
#ifdef _WIN32
    std::wstring pythonHome = PYTHON_HOME;
    SetDllDirectoryW(pythonHome.c_str());
#endif

    if (!Py_IsInitialized())
        Py_Initialize();

#ifdef _WIN32
    PyRun_SimpleString("import win_patch_codecarbon");
#endif

    // Import the shared bridge from the repository root rather than a private
    // copy resolved through the working directory.
    run("import sys");
    run(std::string("sys.path.insert(0, r'") + PROJECT_SOURCE_DIR + "/..')");

    // The embedded interpreter resolves site-packages from whatever libpython
    // was linked against, which is not the environment the rest of the campaign
    // uses -- so codecarbon is simply absent. Add the packages of the
    // interpreter named by the run contract, so this stack measures with the
    // same CodeCarbon build as every other.
    run(std::string(
        "import os, subprocess, sys\n"
        "_py = os.environ.get('DEEPGREEN_PYTHON')\n"
        "if _py and os.path.exists(_py):\n"
        "    _sp = subprocess.run([_py, '-c', 'import site,sys; "
        "print(\\'\\\\n\\'.join(site.getsitepackages()+[p for p in sys.path if p]))'],\n"
        "                         capture_output=True, text=True).stdout.split()\n"
        "    for _p in _sp:\n"
        "        if _p not in sys.path:\n"
        "            sys.path.append(_p)\n"
        "elif _py:\n"
        "    raise RuntimeError('DEEPGREEN_PYTHON points at a missing interpreter: ' + _py)\n"));

    run("from tools import deepgreen_tracker as _dg");
    run("_dg.write_manifest()");
}

void PythonTracker::finalizeTracker()
{
    if (Py_IsInitialized())
        Py_Finalize();
}

void PythonTracker::startTracker(const std::string& phase, const uint32_t epoch)
{
    run("_dg.start('" + phase + "', " + std::to_string(epoch) + ")");
}

void PythonTracker::stopTracker()
{
    run("_dg.stop()");
}

void PythonTracker::logMetric(const uint32_t epoch, const double trainLoss,
                              const double testLoss, const double testAcc)
{
    run("_dg.metric(epoch=" + std::to_string(epoch) +
        ", train_loss=" + std::to_string(trainLoss) +
        ", test_loss=" + std::to_string(testLoss) +
        ", test_acc=" + std::to_string(testAcc) + ")");
}

void PythonTracker::logDataFingerprint(const int64_t n, const double mean,
                                       const double sd, const double min,
                                       const double max)
{
    run("_dg.data_fingerprint(split='test', n=" + std::to_string(n) +
        ", mean=" + std::to_string(mean) +
        ", sd=" + std::to_string(sd) +
        ", min=" + std::to_string(min) +
        ", max=" + std::to_string(max) + ")");
}

PythonTracker::RunParams PythonTracker::runParams()
{
    const uint32_t rep = static_cast<uint32_t>(envAsU64("DEEPGREEN_REP", 0));
    return RunParams{
        rep,
        envAsU64("DEEPGREEN_SEED", 1000 + rep),
        static_cast<uint32_t>(envAsU64("DEEPGREEN_EPOCHS", 30)),
    };
}
