#ifndef PYTHONTRACKER_H
#define PYTHONTRACKER_H
#pragma once
#include <string>


namespace PythonTracker {

    void initializeTracker();

    void startTracker(const std::string& outputDir, const std::string& outputFile);

    void stopTracker();

    void finalizeTracker();

}



#endif //PYTHONTRACKER_H
