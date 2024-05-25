#include <cmath>
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
    registry.registerComponent<AgentType>();
    registry.registerComponent<ChunkInfo>();
    registry.registerComponent<ChunkData>();
    registry.registerComponent<SurroundingObservation>();
    registry.registerComponent<Health>();
    registry.registerComponent<HealthAccumulator>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<ChunkInfoArchetype>();
    registry.registerArchetype<ChunkDataArchetype>();

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

static inline void initWorld(Engine &ctx,
                             uint32_t num_chunks_x,
                             uint32_t num_chunks_y)
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

        ctx.get<AgentType>(entity) = (i == 0) ?
            AgentType::Herbivore : AgentType::Carnivore;

        ctx.get<Health>(entity).v = 100;
        ctx.get<HealthAccumulator>(entity).v.store_relaxed(100);
    }

    ma::Loc loc = ctx.makeStationary<ChunkInfoArchetype>(num_chunks_x * num_chunks_y);

    ctx.data().chunksLoc = loc;
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    // TODO: Implement world resetting
}

inline void resetChunkInfoSystem(Engine &ctx,
                                 ChunkInfo &chunk_info)
{
    uint32_t num_agents = chunk_info.numAgents.load_relaxed();

    chunk_info.numAgents.store_relaxed(0);
    chunk_info.totalSpeed.store_relaxed(0.0f);
}

inline void actionSystem(Engine &ctx,
                         ma::Entity e,
                         ma::base::Rotation &rot,
                         ma::base::Position &pos,
                         AgentType agent_type,
                         Action &action,
                         ma::render::RenderCamera &camera)
{
    // First perform the shoot action if needed - we use the contents of the
    // previous frame from the raycasting output.
    if (action.shoot) {
        ma::render::FinderOutput *finder_output = 
            (ma::render::FinderOutput *)
                ma::render::RenderingSystem::getRenderOutput<
                ma::render::FinderOutputBuffer>(
                        ctx, camera, sizeof(ma::render::FinderOutput));

        if (finder_output->hitEntity != ma::Entity::none()) {
            // Each hit leads to -10 health damage.
            ctx.get<HealthAccumulator>(finder_output->hitEntity).v.
                fetch_add_relaxed(-10);
        }
    }

    if (action.rotate) {
        rot *= ma::math::Quat::angleAxis(
                0.1f, ma::math::Vector3{ 0.f, 0.f, 1.f });
    }

    ma::math::Vector3 old_pos = pos;

    ma::math::Vector3 view_dir = 
        rot.rotateVec(ma::math::Vector3{1.f, 0.f, 0.f});
    view_dir = view_dir.normalize();

    if (action.forward) {
        // Get the view direction
        pos += view_dir;
    } else if (action.backward) {
        pos -= view_dir;
    }

    // Make sure to clamp the position to the world boundaries
    const float kWorldLimitX = ctx.data().cellDim * 
                               (float)ChunkData::kChunkWidth *
                               (float)ctx.data().numChunksX;
    const float kWorldLimitY = ctx.data().cellDim * 
                               (float)ChunkData::kChunkWidth *
                               (float)ctx.data().numChunksY;

    pos.x = std::min(kWorldLimitX, std::max(0.f, pos.x));
    pos.y = std::min(kWorldLimitY, std::max(0.f, pos.y));

    ma::math::Vector3 delta_pos = pos - old_pos;
    float delta_pos_len = delta_pos.length();

    // Update the chunk data for the chunk that this agent affects
    ma::math::Vector2 chunk_coord = ctx.data().getChunkCoord(pos.xy());
    int32_t chunk_idx = ctx.data().getChunkIndex(chunk_coord);

    assert(chunk_idx != -1);

    ChunkInfo *chunk_info = ctx.data().getChunkInfo(ctx, chunk_idx);

    // Increment the number of agents there are in this chunk.
    chunk_info->numAgents.fetch_add_relaxed(1);
    chunk_info->totalSpeed.fetch_add_relaxed((uint32_t)(delta_pos_len*2.f));
}

inline void healthSync(Engine &ctx,
                       ma::Entity e,
                       Health &health,
                       HealthAccumulator &health_accum)
{
    health.v = health_accum.v.load_relaxed();

    LOG("Entity({},{}) has health {}\n", e.gen, e.id, health.v);

    if (health.v <= 0) {
        // Destroy myself!
        ma::render::RenderingSystem::cleanupViewingEntity(ctx, e);
        ctx.destroyRenderableEntity(e);

        LOG("Entity({}, {}) has been destroyed!\n", e.gen, e.id);
    }
}

inline void updateSurroundingObservation(Engine &ctx,
                                         ma::Entity e,
                                         ma::base::Position &pos,
                                         AgentType agent_type,
                                         SurroundingObservation &surroundings)
{
    ma::math::Vector2 cell_pos = pos.xy() / ctx.data().cellDim;
    cell_pos -= ma::math::Vector2{ (float)ChunkData::kChunkWidth * 0.5f,
                                   (float)ChunkData::kChunkWidth * 0.5f };
    ma::math::Vector2 chcoord = cell_pos / (float)ChunkData::kChunkWidth;

    // These are the coordinates of the chunks with centroids which
    // surround this agent.
    ma::math::Vector2 chcoord00,
                      chcoord10,
                      chcoord01,
                      chcoord11;

    chcoord00.x = std::floor(chcoord.x);
    chcoord00.y = std::floor(chcoord.y);

    chcoord10.x = std::ceil(chcoord.x);
    chcoord10.y = std::floor(chcoord.y);

    chcoord01.x = std::floor(chcoord.x);
    chcoord01.y = std::ceil(chcoord.y);

    chcoord11.x = std::ceil(chcoord.x);
    chcoord11.y = std::ceil(chcoord.y);

    int32_t chindex00 = ctx.data().getChunkIndex(chcoord00),
            chindex10 = ctx.data().getChunkIndex(chcoord10),
            chindex01 = ctx.data().getChunkIndex(chcoord01),
            chindex11 = ctx.data().getChunkIndex(chcoord11);

    float x_interpolant = chcoord.x - chcoord00.x;
    float y_interpolant = chcoord.y - chcoord00.y;

    ChunkInfo *chinfo00 = ctx.data().getChunkInfo(ctx, chindex00),
              *chinfo10 = ctx.data().getChunkInfo(ctx, chindex10),
              *chinfo01 = ctx.data().getChunkInfo(ctx, chindex01),
              *chinfo11 = ctx.data().getChunkInfo(ctx, chindex11);

    float num_agents00 = (chinfo00 ? (float)chinfo00->numAgents.load_relaxed() : 0.f),
          num_agents10 = (chinfo10 ? (float)chinfo10->numAgents.load_relaxed() : 0.f),
          num_agents01 = (chinfo01 ? (float)chinfo01->numAgents.load_relaxed() : 0.f),
          num_agents11 = (chinfo11 ? (float)chinfo11->numAgents.load_relaxed() : 0.f);

    float total_speed00 = (chinfo00 ? (float)chinfo00->totalSpeed.load_relaxed() : 0.f),
          total_speed10 = (chinfo10 ? (float)chinfo10->totalSpeed.load_relaxed() : 0.f),
          total_speed01 = (chinfo01 ? (float)chinfo01->totalSpeed.load_relaxed() : 0.f),
          total_speed11 = (chinfo11 ? (float)chinfo11->totalSpeed.load_relaxed() : 0.f);

    float num_agents_x_0 = x_interpolant * num_agents10 + 
                           (1.f - x_interpolant) * num_agents00;
    float num_agents_x_1 = x_interpolant * num_agents11 + 
                           (1.f - x_interpolant) * num_agents01;

    float total_speed_x_0 = x_interpolant * total_speed10 + 
                           (1.f - x_interpolant) * total_speed00;
    float total_speed_x_1 = x_interpolant * total_speed11 + 
                           (1.f - x_interpolant) * total_speed01;

    float num_agents_interpolated = y_interpolant * num_agents_x_1 +
                                    (1.f - y_interpolant) * num_agents_x_0;

    float total_speed_interpolated = y_interpolant * total_speed_x_1 +
                                    (1.f - y_interpolant) * total_speed_x_0;

    surroundings.presenceHeuristic = num_agents_interpolated;
    surroundings.movementHeuristic = total_speed_interpolated;
}

inline void rewardSystem(Engine &,
                         ma::base::Position pos,
                         Reward &out_reward)
{
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
    // Reset the information that tracks the number of agents / movement
    // happening within a chunk
    auto reset_chunk_info = builder.addToGraph<ma::ParallelForNode<Engine,
        resetChunkInfoSystem,
            ChunkInfo
        >>({});

    // Turn policy actions into movement
    auto action_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        actionSystem,
            ma::Entity,
            ma::base::Rotation,
            ma::base::Position,
            AgentType,
            Action,
            ma::render::RenderCamera
        >>({reset_chunk_info});

    auto health_sync_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        healthSync,
            ma::Entity,
            Health,
            HealthAccumulator,
        >>({action_sys});

    auto update_surrounding_obs_sys = builder.addToGraph<ma::ParallelForNode<Engine,
         updateSurroundingObservation,
            ma::Entity,
            ma::base::Position,
            AgentType,
            SurroundingObservation
         >>({health_sync_sys});

    // Conditionally reset the world if the episode is over
    auto reward_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        rewardSystem,
            ma::base::Position,
            Reward
        >>({update_surrounding_obs_sys});

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
      numAgents(cfg.numAgentsPerWorld),
      numChunksX(cfg.numChunksX),
      numChunksY(cfg.numChunksY),
      cellDim(cfg.cellDim)
{
    initWorld(ctx, numChunksX, numChunksY);

    // Initialize state required for the raytracing
    ma::render::RenderingSystem::init(ctx, nullptr);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
