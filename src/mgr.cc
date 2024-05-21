#include "mgr.h"

#include <stdio.h>
#include <madrona/mw_gpu.hpp>

namespace ma = madrona;



// We only support CUDA implementation of the simulator because we need
// raytracing support for agent sensors.
struct Manager::Impl {
    // Encapsulates the GPU executor
    ma::MWCudaExecutor gpuExec;

    // Launches the graph for stepping the simulator.
    ma::MWCudaLaunchGraph stepGraph;

    // Launches the graph for raytracing and raytracing setup.
    ma::MWCudaLaunchGraph sensorGraph;



    static Impl *make(const Manager::Config &cfg);
};

Manager::Impl *Manager::Impl::make(const Config &cfg)
{
    
}




Manager::Manager(const Config &cfg)
    : mImpl(std::unique_ptr<Impl>(Impl::make(cfg)))
{
}

Manager::~Manager()
{
}
