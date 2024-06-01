#pragma once

#include "entry/mgr.hpp"

// This will require initializing just like the manager
struct ScriptBotsViewer : Manager {
    struct Config {
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
        uint32_t initNumAgentsPerWorld; // Number of agents at the 
                                        // start of the simulation

        uint32_t windowWidth;
        uint32_t windowHeight;
    };

    ScriptBotsViewer(const Config &cfg);
    ~ScriptBotsViewer() override;

    void loop();

private:
    struct ViewerImpl;

    ViewerImpl *impl_;
};

