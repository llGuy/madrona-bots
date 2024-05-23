#include <algorithm>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "types.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) ma::mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

namespace ma = madrona;

namespace mbots {

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ma::ECSRegistry &registry, const Config &cfg)
{
    ma::base::registerTypes(registry);

    ma::render::RenderingSystem::registerTypes(registry, nullptr);

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::RenderOutputBuffer>(
            (uint32_t)ExportID::Sensor);
}

static inline void initWorld(Engine &ctx)
{
    for (int i = 0; i < ctx.data().numAgents; ++i) {
        auto entity = ctx.makeRenderableEntity<Agent>();

        // Initialize the entities with some positions
        ctx.get<ma::base::Position>(entity) = ma::math::Vector3{
            i * 10.f, 0.f, 0.f
        };

        ctx.get<ma::base::Rotation>(entity) =
            ma::math::Quat::angleAxis(0.f, ma::math::Vector3{0.f, 0.f, 1.f});

        ctx.get<ma::base::Scale>(entity) = ma::math::Diag3x3{
            1.f, 1.f, 1.f
        };

        // Attach a view to this entity so that sensor data gets generated
        // for it.
        ma::render::RenderingSystem::attachEntityToView(
            ctx, entity, 90.f, 0.1f, { 0.f, 0.f, 0.f });
    }
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    // TODO: Implement world resetting
}

inline void actionSystem(Engine &ctx,
                         ma::Entity e,
                         ma::base::Rotation &rot,
                         ma::base::Position &pos,
                         Action &action)
{
    LOG("Hello from actionSystem!\n");

    // For now, the action is just going to rotate the entities.
    rot *= ma::math::Quat::angleAxis(0.01f, ma::math::Vector3{ 0.f, 0.f, 1.f });
}

inline void rewardSystem(Engine &,
                         ma::base::Position pos,
                         Reward &out_reward)
{
    LOG("Hello from reward system!\n");
}

inline void nopSystem(Engine &,
                      ma::Entity)
{
}

template <typename ArchetypeT>
ma::TaskGraph::NodeID queueSortByWorld(ma::TaskGraph::Builder &builder,
                                       ma::Span<const ma::TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<ma::SortArchetypeNode<ArchetypeT, ma::WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ma::ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

static void setupStepTasks(ma::TaskGraphBuilder &builder, 
                           const Sim::Config &cfg)
{
    // Turn policy actions into movement
    auto action_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        actionSystem,
            ma::Entity,
            ma::base::Rotation,
            ma::base::Position,
            Action,
        >>({});

    // Conditionally reset the world if the episode is over
    auto reward_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        rewardSystem,
            ma::base::Position,
            Reward
        >>({action_sys});

    // Required
    auto clear_tmp = builder.addToGraph<ma::ResetTmpAllocNode>({reward_sys});
    auto recycle_sys = builder.addToGraph<ma::RecycleEntitiesNode>({clear_tmp});
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {recycle_sys});

    (void)sort_agents;
}

static void setupSensorTasks(ma::TaskGraphBuilder &builder, 
                             const Sim::Config &cfg)
{
    // This task graph is also going to perform the tracing
    ma::render::RenderingSystem::setupTasks(builder, {});
}

// Build the task graph
void Sim::setupTasks(ma::TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
    setupSensorTasks(taskgraph_mgr.init(TaskGraphID::Sensor), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx),
      curWorldEpisode(0),
      initRandKey(cfg.initRandKey),
      rng(ma::rand::split_i(initRandKey, curWorldEpisode++,
                            (uint32_t)ctx.worldID().idx)),
      autoReset(false),
      numAgents(cfg.numAgentsPerWorld)
{
    initWorld(ctx);

    // Initialize state required for the raytracing
    ma::render::RenderingSystem::init(ctx, nullptr);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
