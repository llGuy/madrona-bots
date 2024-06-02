#pragma once

#include "entry/mgr.hpp"

#include <functional>

// This will require initializing just like the manager
struct ScriptBotsViewer {
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
    ~ScriptBotsViewer();

    Manager *getManager();

    template <typename StepFn>
    void loop(StepFn &&step_fn)
    {
        loopImpl([](void *data) {
            auto *step_fn_ptr = (StepFn *)data;
            (*step_fn_ptr)();
        }, (void *)&step_fn);
    }

private:
    void loopImpl(void (*step_fn)(void *data), void *step_fn_data);

    struct ViewerImpl;

    ViewerImpl *impl_;
};
