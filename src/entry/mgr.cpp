#include "mgr.hpp"
#include "sim/sim.hpp"

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

    mbots::Action *actionBuffer;

    mbots::SimBridge *simBridge;

    int32_t *agentWorldOffsets;
    int32_t *agentWorldCounts;



    static Impl *make(const Manager::Config &cfg);

    Impl(const Manager::Config &mgr_cfg,
         ma::MWCudaExecutor &&exec,
         ma::MWCudaLaunchGraph &&step,
         ma::MWCudaLaunchGraph &&sensor,
         mbots::Action *action_buffer,
         mbots::SimBridge *sim_bridge);

    ~Impl();

    inline void step()
    {
        gpuExec.run(stepGraph);
        gpuExec.run(sensorGraph);

        // Read the agent world offsets and counts
        cudaMemcpy(agentWorldOffsets, simBridge->agentWorldOffsets, 
                   sizeof(int32_t) * cfg.numWorlds,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(agentWorldCounts, simBridge->agentWorldCounts, 
                   sizeof(int32_t) * cfg.numWorlds,
                   cudaMemcpyDeviceToHost);
    }

    inline ma::py::Tensor exportTensor(mbots::ExportID slot,
        ma::py::TensorElementType type,
        ma::Span<const int64_t> dims) const
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return ma::py::Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};

Manager::Impl::Impl(const Config &mgr_cfg,
                    ma::MWCudaExecutor &&exec,
                    ma::MWCudaLaunchGraph &&step,
                    ma::MWCudaLaunchGraph &&sensor,
                    mbots::Action *action_buffer,
                    mbots::SimBridge *sim_bridge)
    : cfg(mgr_cfg),
      gpuExec(std::move(exec)),
      stepGraph(std::move(step)),
      sensorGraph(std::move(sensor)),
      actionBuffer(action_buffer),
      simBridge(sim_bridge),
      agentWorldOffsets((int32_t *)malloc(sizeof(int32_t) * mgr_cfg.numWorlds)),
      agentWorldCounts((int32_t *)malloc(sizeof(int32_t) * mgr_cfg.numWorlds))
{
}

Manager::Impl *Manager::Impl::make(const Config &mgr_cfg)
{
    mbots::SimBridge *sim_bridge = (mbots::SimBridge *)ma::cu::allocReadback(
            sizeof(mbots::SimBridge));

    // Initialize the GPU executor and launch graphs
    mbots::Sim::Config sim_cfg = {
        .initRandKey = ma::rand::initKey(mgr_cfg.randSeed),
        .numChunksX = 8,
        .numChunksY = 6,
        .cellDim = 1.f,
        .renderBridge = mgr_cfg.renderBridge,
        .simBridge = sim_bridge,
        .totalAllowedFood = 30
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
        .raycastOutputResolution = mgr_cfg.sensorSize,
        .nearSphere = 1.1f,
    };

    ma::CompileConfig compile_cfg = {
        // Defined by the build system
        .userSources = { MBOTS_SRC_LIST },
        .userCompileFlags = { MBOTS_COMPILE_FLAGS },
        .optMode = ma::CompileConfig::OptMode::LTO
    };

    CUcontext cu_ctx = ma::MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
    ma::MWCudaExecutor gpu_exec(state_cfg, compile_cfg, cu_ctx);

    ma::MWCudaLaunchGraph init_graph = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::Init, false);
    ma::MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::Step, false);
    ma::MWCudaLaunchGraph sensor_graph = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::Sensor, true);

    // Run the init taskgraph
    gpu_exec.run(init_graph);

    mbots::Action *action_buffer = (mbots::Action *)
        gpu_exec.getExported((uint32_t)mbots::ExportID::Action);

    return new Impl(mgr_cfg, 
                    std::move(gpu_exec),
                    std::move(step_graph),
                    std::move(sensor_graph),
                    action_buffer,
                    sim_bridge);
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

ma::py::Tensor Manager::sensorTensor() const
{
    uint32_t pixels_per_view = impl_->cfg.sensorSize;
    return impl_->exportTensor(mbots::ExportID::Sensor,
                               ma::py::TensorElementType::UInt8,
                               {
                                   impl_->simBridge->totalNumAgents,
                                   pixels_per_view,
                               });
}


void Manager::setAction(uint32_t agent_idx,
                        int32_t forward,
                        int32_t backward,
                        int32_t rotate,
                        int32_t shoot)
{
    mbots::Action action = {
        .forward = forward,
        .backward = backward,
        .rotate = rotate,
        .shoot = shoot
    };

    auto *action_ptr = impl_->actionBuffer + agent_idx;

    cudaMemcpy(action_ptr, &action, sizeof(mbots::Action),
                cudaMemcpyHostToDevice);
}

uint32_t Manager::agentOffsetForWorld(uint32_t world_idx)
{
    return impl_->agentWorldOffsets[world_idx];
}
