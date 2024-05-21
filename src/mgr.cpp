#include "mgr.hpp"
#include "sim.hpp"

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include <madrona/utils.hpp>

#include <madrona/heap_array.hpp>
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>

namespace ma = madrona;

struct Manager::Impl {
    // Manager's configuration
    Config cfg;

    // Encapsulates state required for invoking the GPU execution engine
    ma::MWCudaExecutor gpuExec;

    // GPU task graphs required for transitioning state/computing rewards
    // and observations.
    ma::MWCudaLaunchGraph stepGraph;
    ma::MWCudaLaunchGraph sensorGraph;



    static Impl *make(const Manager::Config &cfg);

    Impl(const Manager::Config &mgr_cfg,
         ma::MWCudaExecutor &&exec,
         ma::MWCudaLaunchGraph &&step,
         ma::MWCudaLaunchGraph &&sensor);

    ~Impl();

    inline void step()
    {
        gpuExec.run(stepGraph);
        gpuExec.run(sensorGraph);
    }
};

Manager::Impl::Impl(const Config &mgr_cfg,
                    ma::MWCudaExecutor &&exec,
                    ma::MWCudaLaunchGraph &&step,
                    ma::MWCudaLaunchGraph &&sensor)
    : cfg(cfg),
      gpuExec(std::move(exec)),
      stepGraph(std::move(step)),
      sensorGraph(std::move(sensor))
{
}

Manager::Impl *Manager::Impl::make(const Config &mgr_cfg)
{
    // Initialize the GPU executor and launch graphs
    mbots::Sim::Config sim_cfg = {
        .numAgentsPerWorld = 32, // TODO: Allow for finer control over this.
        .initRandKey = ma::rand::initKey(mgr_cfg.randSeed)
    };



    ma::HeapArray<mbots::Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

    ma::StateConfig state_cfg = {
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(mbots::Sim::WorldInit),

        .userConfigPtr = (void *)&sim_cfg,
        .numUserConfigBytes = sizeof(mbots::Sim::Config),

        .numWorldDataBytes = sizeof(mbots::Sim),
        .worldDataAlignment = alignof(mbots::Sim),

        .numWorlds = mgr_cfg.numWorlds,
        .numTaskGraphs = (uint32_t)mbots::TaskGraphID::NumTaskGraphs,
        .numExportedBuffers = (uint32_t)mbots::ExportID::NumExports,
        .nearSphere = 0.01f,
    };

    ma::CompileConfig compile_cfg = {
        // Defined by the build system
        .userSources = { MBOTS_SRC_LIST },
        .userCompileFlags = { MBOTS_COMPILE_FLAGS },
        .optMode = ma::CompileConfig::OptMode::LTO
    };

    CUcontext cu_ctx = ma::MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
    ma::MWCudaExecutor gpu_exec(state_cfg, compile_cfg, cu_ctx);

    ma::MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::Step, false);
    ma::MWCudaLaunchGraph sensor_graph = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::Sensor, false);

    return new Impl(mgr_cfg, 
                    std::move(gpu_exec),
                    std::move(step_graph),
                    std::move(sensor_graph));
}

Manager::Impl::~Impl()
{
}



Manager::Manager(const Config &cfg)
    : impl_(Impl::make(cfg))
{
}

Manager::~Manager()
{
}

void Manager::step()
{
    impl_->step();
}
