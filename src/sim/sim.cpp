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

    // The render bridge is going to hold something when running with the
    // visualizer application.
    ma::render::RenderingSystem::registerTypes(registry, 
            (ma::render::RenderECSBridge *)cfg.renderBridge);

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<AgentType>();
    registry.registerComponent<ChunkInfo>();
    registry.registerComponent<SurroundingObservation>();
    registry.registerComponent<Health>();
    registry.registerComponent<HealthAccumulator>();
    registry.registerComponent<Species>();
    registry.registerComponent<SpeciesObservation>();
    registry.registerComponent<PositionObservation>();
    registry.registerComponent<HealthObservation>();
    registry.registerComponent<AgentObservationBridge>();
    registry.registerComponent<SensorOutputIndex>();

    registry.registerComponent<SpeciesInfoTracker>();
    registry.registerComponent<SpeciesCount>();
    registry.registerComponent<SpeciesReward>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<BridgeSync>();
    registry.registerSingleton<AddFoodSingleton>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<StaticObject>();
    registry.registerArchetype<ChunkInfoArchetype>();
    registry.registerArchetype<AgentObservationArchetype>();
    registry.registerArchetype<SpeciesInfoArchetype>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);

    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::SemanticOutputBuffer>(
            (uint32_t)ExportID::SensorSemantic);
    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::SemanticOutputBuffer>(
            (uint32_t)ExportID::SensorDepth);

    registry.exportColumn<Agent, SensorOutputIndex>(
            (uint32_t)ExportID::SensorIndex);

    registry.exportColumn<SpeciesInfoArchetype, SpeciesCount>(
            (uint32_t)ExportID::SpeciesCount);
    registry.exportColumn<SpeciesInfoArchetype, SpeciesReward>(
            (uint32_t)ExportID::SpeciesReward);
}

static inline void makeFloorPlane(Engine &ctx,
                                  uint32_t num_chunks_x,
                                  uint32_t num_chunks_y)
{
    auto floor_plane = ctx.makeRenderableEntity<StaticObject>();

    ctx.get<ma::base::Position>(floor_plane) = ma::math::Vector3 {
        0.f, 0.f, 0.f
    };

    ctx.get<ma::base::Rotation>(floor_plane) = ma::math::Quat {
        1.f, 0.f, 0.f, 0.f
    };

    ctx.get<ma::base::Scale>(floor_plane) = ma::math::Diag3x3{
        1.f, 1.f, 1.f
    };

    ctx.get<ma::base::ObjectID>(floor_plane).idx = (int32_t)SimObject::Plane;
}

static inline void makeWalls(Engine &ctx,
                             uint32_t num_chunks_x,
                             uint32_t num_chunks_y)
{
    float world_len_x = num_chunks_x *
                        ChunkInfo::kChunkWidth *
                        ctx.data().cellDim;
    float world_len_y = num_chunks_y *
                        ChunkInfo::kChunkWidth *
                        ctx.data().cellDim;

    ma::math::Vector3 wall_centroids[4] = {
        { world_len_x * 0.5f, 0.f, 0.f },
        { 0.f, world_len_y * 0.5f, 0.f },
        { world_len_x * 0.5f, world_len_y, 0.f },
        { world_len_x, world_len_y * 0.5f, 0.f },
    };

    ma::math::Diag3x3 wall_scales[4] = {
        { world_len_x * 0.5f, 0.2f, 1.f },
        { 0.2f, world_len_y * 0.5f, 1.f },
        { world_len_x * 0.5f, 0.2f, 1.f },
        { 0.2f, world_len_y * 0.5f, 1.f },
    };

    for (int i = 0; i < 4; ++i) {
        auto floor_plane = ctx.makeRenderableEntity<StaticObject>();
        ctx.get<ma::base::Position>(floor_plane) = wall_centroids[i];

        ctx.get<ma::base::Rotation>(floor_plane) = ma::math::Quat {
            1.f, 0.f, 0.f, 0.f
        };

        ctx.get<ma::base::Scale>(floor_plane) = wall_scales[i];

        ctx.get<ma::base::ObjectID>(floor_plane).idx = (int32_t)SimObject::Wall;
    }
}

static inline ma::Entity makeAgent(Engine &ctx,
                                   const ma::math::Vector3 &pos,
                                   uint32_t species_id,
                                   uint32_t initial_health)
{
    auto entity = ctx.makeAgent();

    // Initialize the entities with some positions
    ctx.get<ma::base::Position>(entity) = pos;

    ctx.get<ma::base::Rotation>(entity) =
        ma::math::Quat::angleAxis(0.f, ma::math::Vector3{0.f, 0.f, 1.f});

    ctx.get<ma::base::Scale>(entity) = ma::math::Diag3x3{
        1.f, 1.f, 1.f
    };

    ctx.get<ma::base::ObjectID>(entity).idx = (int32_t)SimObject::Agent;

    // Attach a view to this entity so that sensor data gets generated
    // for it.
    ma::render::RenderingSystem::attachEntityToView(
            ctx, entity, 90.f, 0.1f, { 0.f, 0.f, 0.f });

    ctx.get<AgentType>(entity) = AgentType::Carnivore;

    ctx.get<Health>(entity).v = initial_health;
    ctx.get<HealthAccumulator>(entity).v.store_relaxed(initial_health);

    ctx.get<Species>(entity).speciesID = species_id;

    return entity;
}

static inline void initWorld(Engine &ctx,
                             uint32_t num_chunks_x,
                             uint32_t num_chunks_y)
{
    makeFloorPlane(ctx, num_chunks_x, num_chunks_y);
    makeWalls(ctx, num_chunks_x, num_chunks_y);

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            auto entity = makeAgent(
                    ctx,
                    ma::math::Vector3{ 3.f + x * 10.f, 3.f + y * 10.f, 1.f },
                    x + 1,
                    100);

            (void)entity;
        }
    }

    ma::Loc loc = ctx.makeStationary<ChunkInfoArchetype>(
            num_chunks_x * num_chunks_y);

    ctx.data().chunksLoc = loc;



    auto species_tracker = ctx.makeEntity<SpeciesInfoArchetype>();

    for (int i = 0; i < kNumSpecies; ++i) {
        ctx.get<SpeciesInfoTracker>(species_tracker).countTracker[i].store_relaxed(0);
        ctx.get<SpeciesInfoTracker>(species_tracker).healthTracker[i].store_relaxed(0);
    }

    ctx.data().speciesInfoTracker = species_tracker;
}

inline void initializeChunks(Engine &ctx,
                             ChunkInfo &chunk_info)
{
    auto *state_mgr = ma::mwGPU::getStateManager();

    ChunkInfo *base = state_mgr->getArchetypeComponent<
        ChunkInfoArchetype, ChunkInfo>() + ctx.data().chunksLoc.row;

    uint32_t linear_idx = &chunk_info - base;

    chunk_info.chunkCoord.x = linear_idx % ctx.data().numChunksX;
    chunk_info.chunkCoord.y = linear_idx / ctx.data().numChunksX;

    for (int i = 0; i < ChunkInfo::kMaxFoodPackages; ++i) {
        chunk_info.foodPackages[i].numFood.store_relaxed(0);
        chunk_info.foodPackages[i].x = 0;
        chunk_info.foodPackages[i].y = 0;
    }

#if 0
    memset(chunk_info.data, 0, 
           sizeof(uint8_t) * ChunkInfo::kChunkWidth * ChunkInfo::kChunkWidth);
#endif
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    // TODO: Implement world resetting
}

// Returns true if food was added successfully
inline bool addFoodToChunk(Engine &ctx,
                           ChunkInfo &chunk_info)
{
    uint32_t rand_x = ctx.data().rng.sampleI32(0, ChunkInfo::kChunkWidth);
    uint32_t rand_y = ctx.data().rng.sampleI32(0, ChunkInfo::kChunkWidth);

    for (int i = 0; i < ChunkInfo::kMaxFoodPackages; ++i) {
        FoodPackage &food_pkg = chunk_info.foodPackages[i];

        // If this food package has nothing, 
        if (food_pkg.numFood.load_relaxed() == 0) {
            uint32_t rand_x = ctx.data().rng.sampleI32(
                    0, ChunkInfo::kChunkWidth);
            uint32_t rand_y = ctx.data().rng.sampleI32(
                    0, ChunkInfo::kChunkWidth);

            food_pkg.x = rand_x;
            food_pkg.y = rand_y;

            food_pkg.numFood.store_relaxed(1);

            { // For visualization purposes, we also add a food entity
                auto food_ent = ctx.makeRenderableEntity<StaticObject>();

                ctx.get<ma::base::Position>(food_ent) = ma::math::Vector3 {
                    ((float)rand_x + chunk_info.chunkCoord.x * ChunkInfo::kChunkWidth),
                    ((float)rand_y + chunk_info.chunkCoord.y * ChunkInfo::kChunkWidth),
                    1.f
                } * ctx.data().cellDim;

                ctx.get<ma::base::Rotation>(food_ent) = 
                    ma::math::Quat::angleAxis(
                            2.f * ma::math::pi * ctx.data().rng.sampleUniform(),
                            ma::math::Vector3{ 0.f, 0.f, 1.f });

                ctx.get<ma::base::Scale>(food_ent) = ma::math::Diag3x3{
                    1.f, 1.f, 1.f
                };

                ctx.get<ma::base::ObjectID>(food_ent).idx = (int32_t)SimObject::Food;

                food_pkg.foodEntity = food_ent;
            }

            // Only break if we add a NEW food item.
            return true;
        } else if (food_pkg.numFood.load_relaxed() < 
                ChunkInfo::kMaxFoodPerPackage) {
            food_pkg.numFood.fetch_add_relaxed(1);
        }
    }

    return false;
}

inline void addFoodSystem(Engine &ctx,
                          AddFoodSingleton)
{
    if (ctx.data().rng.sampleI32(0, 10) == 0) {
        uint32_t rand_sample = ctx.data().rng.sampleI32(1, 3);

        uint32_t diff_allowed = ctx.data().totalAllowedFood -
            ctx.data().currentNumFood.load_relaxed();

        rand_sample = std::min(rand_sample, diff_allowed);

        for (int i = 0; i < rand_sample; ++i) {
            uint32_t chunk_x = ctx.data().rng.sampleI32(0, ctx.data().numChunksX);
            uint32_t chunk_y = ctx.data().rng.sampleI32(0, ctx.data().numChunksY);

            uint32_t linear_idx = chunk_x + chunk_y * ctx.data().numChunksX;

            ChunkInfo *chunk_info = ctx.data().getChunkInfo(ctx, linear_idx);

            if (addFoodToChunk(ctx, *chunk_info)) {
                ctx.data().currentNumFood.fetch_add_relaxed(1);
            }
        }
    }
}

// Here, we will also do food adding
inline void resetChunkInfoSystem(Engine &ctx,
                                 ChunkInfo &chunk_info)
{
    uint32_t num_agents = chunk_info.numAgents.load_relaxed();

    chunk_info.numAgents.store_relaxed(0);
    chunk_info.totalSpeed.store_relaxed(0.0f);


#if 0

    // Randomly decide on whether to add food in this chunk
    uint32_t rand_sample = ctx.data().rng.sampleI32(0, 100);

    if (rand_sample <= 1) {
        // Create a chunk data struct if needed and add food
        // This might go over the total allowed food but doesn't matter,
        // it'll not go over too much
        if (ctx.data().currentNumFood.load_relaxed() + 1 < 
            ctx.data().totalAllowedFood) {

            if (addFoodToChunk(ctx, chunk_info)) {
                ctx.data().currentNumFood.fetch_add_relaxed(1);
            }
        }
    }
#endif
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

    if (action.rotateLeft) {
        rot *= ma::math::Quat::angleAxis(
                0.1f, ma::math::Vector3{ 0.f, 0.f, 1.f });
    } else if (action.rotateRight) {
        rot *= ma::math::Quat::angleAxis(
                -0.1f, ma::math::Vector3{ 0.f, 0.f, 1.f });
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
                               (float)ChunkInfo::kChunkWidth *
                               (float)ctx.data().numChunksX;
    const float kWorldLimitY = ctx.data().cellDim * 
                               (float)ChunkInfo::kChunkWidth *
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

// This also takes care of eating food
inline void healthSync(Engine &ctx,
                       ma::Entity e,
                       ma::base::Position &pos,
                       Health &health,
                       HealthAccumulator &health_accum,
                       Action &action,
                       Species &species,
                       ma::render::RenderCamera &camera)
{
    health.v = health_accum.v.load_relaxed();

    { // See whether or not we ate food.
        ma::math::Vector2 cell_pos = pos.xy() / ctx.data().cellDim;
        ma::math::Vector2 chcoord = cell_pos / (float)ChunkInfo::kChunkWidth;

        // Get the x,y cell of inside the chunk
        uint8_t x = ChunkInfo::kChunkWidth * (chcoord.x - std::floor(chcoord.x));
        uint8_t y = ChunkInfo::kChunkWidth * (chcoord.y - std::floor(chcoord.y));

        int32_t chindex = ctx.data().getChunkIndex(chcoord);

        ChunkInfo *chinfo = ctx.data().getChunkInfo(ctx, chindex);

        // Loop through all food packages in the chunk check overlap
        for (int i = 0; i < ChunkInfo::kMaxFoodPackages; ++i) {
            FoodPackage &food_pkg = chinfo->foodPackages[i];

            if (food_pkg.x == x && food_pkg.y == y) {
                // If we managed to consume, break and update health
                if (food_pkg.consume(ctx)) {
                    health.v += 20.f;

                    break;
                }
            }
        }
    }

    // If you choose to breed, you lose 40 health points
    if (action.breed && health.v > 60) {
        ma::render::FinderOutput *finder_output = 
            (ma::render::FinderOutput *)
                ma::render::RenderingSystem::getRenderOutput<
                ma::render::FinderOutputBuffer>(
                        ctx, camera, sizeof(ma::render::FinderOutput));

        if (finder_output->hitEntity != ma::Entity::none()) {
            // Make sure that this entity is part of our species
            if (ctx.get<Species>(finder_output->hitEntity).speciesID ==
                    species.speciesID) {
                health.v -= 40;

                // Make a new entity which starts at 50 health.
                auto entity = makeAgent(ctx, 
                                        ma::math::Vector3{pos.x, pos.y, pos.z},
                                        species.speciesID,
                                        50);
            }
        }
    }

    if (health.v <= 0) {
        // Destroy myself!
        ma::render::RenderingSystem::cleanupViewingEntity(ctx, e);
        ctx.destroyRenderableEntity(e);

        LOG("Entity({}, {}) has been destroyed!\n", e.gen, e.id);
    }

    health_accum.v.store_relaxed(health.v);
}

inline void updateSurroundingObservation(Engine &ctx,
                                         ma::Entity e,
                                         ma::base::Position &pos,
                                         AgentType agent_type,
                                         SurroundingObservation &surroundings)
{
    ma::math::Vector2 cell_pos = pos.xy() / ctx.data().cellDim;
    cell_pos -= ma::math::Vector2{ (float)ChunkInfo::kChunkWidth * 0.5f,
                                   (float)ChunkInfo::kChunkWidth * 0.5f };
    ma::math::Vector2 chcoord = cell_pos / (float)ChunkInfo::kChunkWidth;

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

inline void rewardSystem(Engine &ctx,
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

template <typename ArchetypeT>
ma::TaskGraph::NodeID queueSortBySpecies(ma::TaskGraph::Builder &builder,
                                         ma::Span<const ma::TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<ma::SortArchetypeNode<ArchetypeT, SpeciesObservation>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ma::ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

inline void updateObservations(Engine &ctx,
                               ma::Entity e,
                               AgentObservationBridge &bridge,
                               Species &species,
                               ma::base::Position &pos,
                               Health &health,
                               SurroundingObservation &sur)
{
    ma::Entity obs_e = bridge.obsEntity;

    ctx.get<SpeciesObservation>(obs_e).speciesID = species.speciesID;

    ctx.get<PositionObservation>(obs_e).pos = ma::math::Vector2 {
        pos.x, pos.y
    };

    ctx.get<HealthObservation>(obs_e).v = health.v;

    ctx.get<SurroundingObservation>(obs_e) = {
        sur.presenceHeuristic,
        sur.movementHeuristic
    };
}

inline void updateSensorOutputIdx(Engine &ctx,
                                  Species &species,
                                  AgentObservationBridge bridge,
                                  SensorOutputIndex &output_idx,
                                  ma::render::RenderCamera &cam,
                                  ma::render::Renderable &renderable,
                                  Health &health)
{
    uint32_t row_idx = ctx.loc(bridge.obsEntity).row;

    ctx.get<ma::render::PerspectiveCameraData>(
            cam.cameraEntity).rowIDX = row_idx;

    output_idx.idx = row_idx;

    // Update the species index too
    ctx.get<ma::render::InstanceData>(
            renderable.renderEntity).speciesIDX = species.speciesID;



    // Update the number of agents in given species
    auto species_tracker = ctx.data().speciesInfoTracker;

    auto &info_tracker = ctx.get<SpeciesInfoTracker>(species_tracker);

    info_tracker.countTracker[species.speciesID-1].fetch_add_relaxed(1);
    info_tracker.healthTracker[species.speciesID-1].fetch_add_relaxed(health.v);
}

inline void speciesInfoSync(Engine &ctx,
                            SpeciesInfoTracker &tracker,
                            SpeciesCount &counts,
                            SpeciesReward &rewards)
{
    for (int i = 0; i < kNumSpecies; ++i) {
        uint32_t count = tracker.countTracker[i].load_relaxed();
        uint32_t total_health = tracker.healthTracker[i].load_relaxed();

        float health_avg = (float)total_health / (float)count;
        if (count == 0) {
            health_avg = 0;
        }

        counts.counts[i] = count;
        
        // (100 is the starting health of all agents)
        rewards.rewards[i] = (float)count / 256.0f +
                             health_avg / 100.0f;

        LOG("World={}; Species={}; Count={}; HealthAvg={}; Reward={}\n",
                ctx.worldID().idx,
                i,
                count,
                health_avg,
                rewards.rewards[i]);

        tracker.countTracker[i].store_relaxed(0);
        tracker.healthTracker[i].store_relaxed(0);
    }
}

inline void bridgeSyncSystem(Engine &ctx,
                       BridgeSync)
{
    // Only do this for the first world
    if (ctx.worldID().idx == 0) {
        auto *state_mgr = ma::mwGPU::getStateManager();

        ctx.data().simBridge->totalNumAgents = 
            state_mgr->getArchetypeNumRows<Agent>();
        ctx.data().simBridge->agentWorldOffsets =
            state_mgr->getArchetypeWorldOffsets<Agent>();
        ctx.data().simBridge->agentWorldCounts =
            state_mgr->getArchetypeWorldCounts<Agent>();
    }
}

static void setupInitTasks(ma::TaskGraphBuilder &builder,
                           const Sim::Config &cfg)
{
    auto init_chunks = builder.addToGraph<ma::ParallelForNode<Engine,
         initializeChunks,
            ChunkInfo
         >>({});

    (void)init_chunks;
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

    auto add_food_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        addFoodSystem,
            AddFoodSingleton
        >>({reset_chunk_info});

    // Turn policy actions into movement
    auto action_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        actionSystem,
            ma::Entity,
            ma::base::Rotation,
            ma::base::Position,
            AgentType,
            Action,
            ma::render::RenderCamera
        >>({add_food_sys});

    auto health_sync_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        healthSync,
            ma::Entity,
            ma::base::Position,
            Health,
            HealthAccumulator,
            Action,
            Species,
            ma::render::RenderCamera
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

    auto sort_agents = queueSortByWorld<Agent>(
        builder, {clear_tmp});

    auto recycle_sys = builder.addToGraph<ma::RecycleEntitiesNode>({sort_agents});

    // After the entities get deleted and recycled, we go through
    // all the observation entities to update them.
    auto update_obs_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        updateObservations,
            ma::Entity,
            AgentObservationBridge,
            Species,
            ma::base::Position,
            Health,
            SurroundingObservation
        >>({recycle_sys});

    // Now, sort all the observation entities by species (even across the worlds)
    auto sort_obs = queueSortBySpecies<AgentObservationArchetype>(
            builder, {update_obs_sys});

    // After the sort, we update the perspective camera data row idx output index
    auto update_sensor_idx = builder.addToGraph<ma::ParallelForNode<Engine,
        updateSensorOutputIdx,
            Species,
            AgentObservationBridge,
            SensorOutputIndex,
            ma::render::RenderCamera,
            ma::render::Renderable,
            Health
        >>({sort_obs});

    auto species_info_sys = builder.addToGraph<ma::ParallelForNode<Engine,
         speciesInfoSync,
            SpeciesInfoTracker,
            SpeciesCount,
            SpeciesReward
        >>({update_sensor_idx});

    auto sort_species = queueSortByWorld<SpeciesInfoArchetype>(
            builder, {species_info_sys});

    auto bridge_sync_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        bridgeSyncSystem,
            BridgeSync
        >>({sort_species});

    (void)bridge_sync_sys;
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
    setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
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
      numChunksX(cfg.numChunksX),
      numChunksY(cfg.numChunksY),
      cellDim(cfg.cellDim),
      simBridge(cfg.simBridge),
      totalAllowedFood(cfg.totalAllowedFood),
      currentNumFood(0)
{
    initWorld(ctx, numChunksX, numChunksY);

    // Initialize state required for the raytracing
    ma::render::RenderingSystem::init(ctx,
            (ma::render::RenderECSBridge *)cfg.renderBridge);

    currentNumFood.store_relaxed(0);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
