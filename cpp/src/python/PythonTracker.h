#ifndef PYTHONTRACKER_H
#define PYTHONTRACKER_H
#pragma once
#include <cstdint>
#include <string>

// Bridge to the shared measurement helper, tools/deepgreen_tracker.py.
//
// The first campaign imported a private copy under cpp/py_script/tracker/ that
// constructed EmissionsTracker with its own defaults, while the Rust, R, Java
// and MATLAB stacks each did the same with different settings: five copies,
// four configurations. Every ecosystem now shares one bridge and one run
// contract, so the instrument is identical by construction.
namespace PythonTracker {

    /// Repetition index, seed and epoch count from the shared run contract.
    struct RunParams {
        uint32_t repetition;
        uint64_t seed;
        uint32_t epochs;
    };

    void initializeTracker();

    /// Open the tracked window for one epoch of one phase ("train" or "eval").
    void startTracker(const std::string& phase, uint32_t epoch);

    void stopTracker();

    /// Record per-epoch model quality. The first campaign computed accuracy in
    /// this stack and only printed it, so energy could never be normalised by
    /// the useful work produced.
    void logMetric(uint32_t epoch, double trainLoss, double testLoss, double testAcc);

    RunParams runParams();

    void finalizeTracker();

}

#endif //PYTHONTRACKER_H
