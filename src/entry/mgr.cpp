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
    ma::MWCudaLaunchGraph shiftObsGraph;

    mbots::Action *actionBuffer;

    mbots::SimBridge *simBridge;

    int32_t *agentWorldOffsets;
    int32_t *agentWorldCounts;



    static Impl *make(const Manager::Config &cfg);

    Impl(const Manager::Config &mgr_cfg,
         ma::MWCudaExecutor &&exec,
         ma::MWCudaLaunchGraph &&step,
         ma::MWCudaLaunchGraph &&sensor,
         ma::MWCudaLaunchGraph &&shift_obs_graph,
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

    inline void shiftObservations()
    {
        gpuExec.run(shiftObsGraph);
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
                    ma::MWCudaLaunchGraph &&shift_obs,
                    mbots::Action *action_buffer,
                    mbots::SimBridge *sim_bridge)
    : cfg(mgr_cfg),
      gpuExec(std::move(exec)),
      stepGraph(std::move(step)),
      sensorGraph(std::move(sensor)),
      shiftObsGraph(std::move(shift_obs)),
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
        .totalAllowedFood = 60,
        .initNumAgentsPerWorld = mgr_cfg.initNumAgentsPerWorld
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

    const char *user_srcs[] = {
        MBOTS_SRC_LIST
    };

    ma::CompileConfig compile_cfg = {
        // Defined by the build system
        .userSources = ma::Span(user_srcs, 1),
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
    ma::MWCudaLaunchGraph shift_obs = gpu_exec.buildLaunchGraph(
            mbots::TaskGraphID::ShiftObservations, false);

    // Run the init taskgraph
    gpu_exec.run(init_graph);

    mbots::Action *action_buffer = (mbots::Action *)
        gpu_exec.getExported((uint32_t)mbots::ExportID::Action);

    return new Impl(mgr_cfg, 
                    std::move(gpu_exec),
                    std::move(step_graph),
                    std::move(sensor_graph),
                    std::move(shift_obs),
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

void Manager::shiftObservations()
{
    impl_->shiftObservations();
}

ma::py::Tensor Manager::semanticTensor(bool is_prev) const
{
    uint32_t pixels_per_view = impl_->cfg.sensorSize;

    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevSensorSemantic,
                                   ma::py::TensorElementType::Int8,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       pixels_per_view,
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::SensorSemantic,
                                   ma::py::TensorElementType::Int8,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       pixels_per_view,
                                   });
    }
}

ma::py::Tensor Manager::depthTensor(bool is_prev) const
{
    uint32_t pixels_per_view = impl_->cfg.sensorSize;

    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevSensorDepth,
                                   ma::py::TensorElementType::UInt8,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       pixels_per_view,
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::SensorDepth,
                                   ma::py::TensorElementType::UInt8,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       pixels_per_view,
                                   });
    }
}

ma::py::Tensor Manager::sensorIndexTensor() const
{
    return impl_->exportTensor(mbots::ExportID::SensorIndex,
                               ma::py::TensorElementType::Int32,
                               {
                                   impl_->simBridge->totalNumAgents,
                                   1
                               });
}

void Manager::setAction(uint32_t agent_idx,
                        int32_t forward,
                        int32_t backward,
                        int32_t rotate_left,
                        int32_t rotate_right,
                        int32_t shoot,
                        int32_t breed)
{
    mbots::Action action = {
        .forward = forward,
        .backward = backward,
        .rotateLeft = rotate_left,
        .rotateRight = rotate_right,
        .shoot = shoot,
        .breed = breed
    };

    auto *action_ptr = impl_->actionBuffer + agent_idx;

    cudaMemcpy(action_ptr, &action, sizeof(mbots::Action),
                cudaMemcpyHostToDevice);
}

uint32_t Manager::agentOffsetForWorld(uint32_t world_idx)
{
    return impl_->agentWorldOffsets[world_idx];
}

// One reward per species.
ma::py::Tensor Manager::rewardTensor(bool is_prev) const
{
    // Returns a (total_num_agents, 1) tensor for rewards
    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevReward,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       1
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::Reward,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       1
                                   });
    }
}

ma::py::Tensor Manager::speciesCountTensor() const
{
    return impl_->exportTensor(mbots::ExportID::SpeciesCount,
                               ma::py::TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   mbots::kNumSpecies
                               });
}

ma::py::Tensor Manager::positionTensor(bool is_prev) const
{
    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevPosition,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       2
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::Position,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       2
                                   });
    }
}

ma::py::Tensor Manager::healthTensor(bool is_prev) const
{
    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevHealth,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       1
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::Health,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       1
                                   });
    }
}

ma::py::Tensor Manager::surroundingTensor(bool is_prev) const
{
    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevSurrounding,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       2 // Presence and movement heuristics
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::Surrounding,
                                   ma::py::TensorElementType::Float32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       2 // Presence and movement heuristics
                                   });
    }
}

ma::py::Tensor Manager::actionTensor(bool is_prev) const
{
    if (is_prev) {
        return impl_->exportTensor(mbots::ExportID::PrevAction,
                                   ma::py::TensorElementType::Int32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       6 // There are 6 possible actions to take
                                   });
    } else {
        return impl_->exportTensor(mbots::ExportID::Action,
                                   ma::py::TensorElementType::Int32,
                                   {
                                       impl_->simBridge->totalNumAgents,
                                       6 // There are 6 possible actions to take
                                   });
    }
}
