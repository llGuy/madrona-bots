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
    registry.registerComponent<AgentStats>();

    registry.registerComponent<PrevSpeciesObservation>();
    registry.registerComponent<PrevPositionObservation>();
    registry.registerComponent<PrevHealthObservation>();
    registry.registerComponent<PrevSurroundingObservation>();
    registry.registerComponent<PrevReward>();
    registry.registerComponent<PrevAction>();

    registry.registerComponent<SpeciesInfoTracker>();
    registry.registerComponent<SpeciesCount>();
    registry.registerComponent<SpeciesReward>();

    registry.registerComponent<StatsObservation>();
    registry.registerComponent<PrevStatsObservation>();

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

    registry.exportColumn<AgentObservationArchetype, PositionObservation>(
        (uint32_t)ExportID::Position);
    registry.exportColumn<AgentObservationArchetype, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<AgentObservationArchetype, SurroundingObservation>(
        (uint32_t)ExportID::Surrounding);
    registry.exportColumn<AgentObservationArchetype, HealthObservation>(
        (uint32_t)ExportID::Health);

    registry.exportColumn<AgentObservationArchetype, PrevPositionObservation>(
        (uint32_t)ExportID::PrevPosition);
    registry.exportColumn<AgentObservationArchetype, PrevAction>(
        (uint32_t)ExportID::PrevAction);
    registry.exportColumn<AgentObservationArchetype, PrevSurroundingObservation>(
        (uint32_t)ExportID::PrevSurrounding);
    registry.exportColumn<AgentObservationArchetype, PrevHealthObservation>(
        (uint32_t)ExportID::PrevHealth);

    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);

    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::SemanticOutputBuffer>(
            (uint32_t)ExportID::SensorSemantic);

    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::SemanticOutputBuffer>(
            (uint32_t)ExportID::SensorDepth);

    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::PrevSemanticOutputBuffer>(
            (uint32_t)ExportID::PrevSensorSemantic);

    registry.exportColumn<ma::render::RaycastOutputArchetype,
        ma::render::PrevSemanticOutputBuffer>(
            (uint32_t)ExportID::PrevSensorDepth);

    registry.exportColumn<Agent, SensorOutputIndex>(
            (uint32_t)ExportID::SensorIndex);

    registry.exportColumn<SpeciesInfoArchetype, SpeciesCount>(
            (uint32_t)ExportID::SpeciesCount);

    registry.exportColumn<AgentObservationArchetype, Reward>(
            (uint32_t)ExportID::Reward);
    registry.exportColumn<AgentObservationArchetype, PrevReward>(
            (uint32_t)ExportID::PrevReward);

    registry.exportColumn<AgentObservationArchetype, StatsObservation>(
            (uint32_t)ExportID::Stats);
    registry.exportColumn<AgentObservationArchetype, PrevStatsObservation>(
            (uint32_t)ExportID::PrevStats);
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

    ctx.get<ma::base::ObjectID>(entity) = {
        (int32_t)SimObject::Agent,
        (int32_t)species_id
    };

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
                             uint32_t num_chunks_y,
                             uint32_t init_num_agents)
{
    makeFloorPlane(ctx, num_chunks_x, num_chunks_y);
    makeWalls(ctx, num_chunks_x, num_chunks_y);

    float world_lim_x = num_chunks_x * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;
    float world_lim_y = num_chunks_y * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;

    for (uint32_t i = 0; i < init_num_agents; ++i) {
        // uint32_t species_idx = ctx.data().rng.sampleI32(0, kNumSpecies) + 1;
        uint32_t species_idx = (i % kNumSpecies) + 1;

        float x_pos = ctx.data().rng.sampleUniform() * world_lim_x;
        float y_pos = ctx.data().rng.sampleUniform() * world_lim_y;

        auto entity = makeAgent(
                ctx,
                ma::math::Vector3{ x_pos, y_pos, 1.f },
                species_idx,
                100);
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
                         ma::render::RenderCamera &camera,
                         AgentObservationBridge &obs_bridge,
                         Species &species,
                         AgentStats &agent_stats)
{
    action = ctx.get<Action>(obs_bridge.obsEntity);

    // First perform the shoot action if needed - we use the contents of the
    // previous frame from the raycasting output.
    if (action.shoot) {
        ma::render::FinderOutput *finder_output = 
            (ma::render::FinderOutput *)
                ma::render::RenderingSystem::getRenderOutput<
                ma::render::FinderOutputBuffer>(
                        ctx, camera, sizeof(ma::render::FinderOutput));

        if (finder_output->hitEntity != ma::Entity::none()) {
            // Each hit leads to -50 health damage.
            ctx.get<HealthAccumulator>(finder_output->hitEntity).v.
                fetch_add_relaxed(-50);

            auto &other_species = ctx.get<Species>(finder_output->hitEntity);

            if (other_species.speciesID == species.speciesID) {
                agent_stats.hitFriendlyAgent = 1;
            } else {
                agent_stats.hitEnemyAgent = 1;
            }
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

    pos.x = std::min(kWorldLimitX - 1.0f, std::max(0.f, pos.x));
    pos.y = std::min(kWorldLimitY - 1.0f, std::max(0.f, pos.y));

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
                       ma::render::RenderCamera &camera,
                       AgentStats &agent_stats)
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

                    agent_stats.ateFood = 1;

                    break;
                }
            }
        }
    }

    // If you choose to breed, you lose 40 health points
    if (action.breed && health.v > 10) {
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

                agent_stats.reproduced = 1;
            }
        }
    }

    // Each agent loses 5 health per tick
    // health.v -= 1;

    if (health.v <= 0) {
        // Destroy myself!
        ma::render::RenderingSystem::cleanupViewingEntity(ctx, e);
        ctx.destroyRenderableEntity(e);
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
                               SurroundingObservation &sur,
                               AgentStats &stats)
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

    ctx.get<StatsObservation>(obs_e) = {
        stats.hitFriendlyAgent,
        stats.hitEnemyAgent,
        stats.ateFood,
        stats.reproduced
    };
}

inline void speciesTrackerUpdate(Engine &ctx,
                                 ma::Entity e,
                                 AgentObservationBridge &bridge,
                                 Species species,
                                 Health health)
{
    (void)e, (void)bridge;

    // Update the number of agents in given species
    auto species_tracker = ctx.data().speciesInfoTracker;

    auto &info_tracker = ctx.get<SpeciesInfoTracker>(species_tracker);

    info_tracker.countTracker[species.speciesID-1].fetch_add_relaxed(1);
    info_tracker.healthTracker[species.speciesID-1].fetch_add_relaxed(health.v);
}

inline void updateSensorOutputIdx(Engine &ctx,
                                  Species &species,
                                  AgentObservationBridge bridge,
                                  SensorOutputIndex &output_idx,
                                  ma::render::RenderCamera &cam,
                                  ma::render::Renderable &renderable,
                                  Health &health)
{
    ma::StateManager *mgr = ma::mwGPU::getStateManager();



    auto semantic_output_buffer = (uint8_t *)mgr->getArchetypeComponent<
        ma::render::RaycastOutputArchetype,
        ma::render::SemanticOutputBuffer>();

    auto depth_output_buffer = (uint8_t *)mgr->getArchetypeComponent<
        ma::render::RaycastOutputArchetype,
        ma::render::DepthOutputBuffer>();

    auto prev_semantic_output_buffer = (uint8_t *)mgr->getArchetypeComponent<
        ma::render::RaycastOutputArchetype,
        ma::render::PrevSemanticOutputBuffer>();

    auto prev_depth_output_buffer = (uint8_t *)mgr->getArchetypeComponent<
        ma::render::RaycastOutputArchetype,
        ma::render::PrevDepthOutputBuffer>();



    auto &pres_cam_data = ctx.get<ma::render::PerspectiveCameraData>(
            cam.cameraEntity);

    // We need to copy the contents of the sensor output from the output
    // frame.
    uint32_t prev_row_idx = output_idx.idx;
    uint32_t current_row_idx = ctx.loc(bridge.obsEntity).row;

    memcpy(prev_semantic_output_buffer + current_row_idx * 32,
           semantic_output_buffer + prev_row_idx * 32,
           32);

    memcpy(prev_depth_output_buffer + current_row_idx * 32,
           depth_output_buffer + prev_row_idx * 32,
           32);


    pres_cam_data.rowIDX = current_row_idx;
    output_idx.idx = current_row_idx;

    // Update the species index too
    ctx.get<ma::render::InstanceData>(
            renderable.renderEntity).speciesIDX = species.speciesID;
}

inline void speciesInfoSync(Engine &ctx,
                            SpeciesInfoTracker &tracker,
                            SpeciesCount &counts,
                            SpeciesReward &rewards)
{
    float world_lim_x = ctx.data().numChunksX * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;
    float world_lim_y = ctx.data().numChunksY * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;

    uint32_t init_num_agents_per_species = ctx.data().initNumAgentsPerWorld /
        kNumSpecies;

    for (int i = 0; i < kNumSpecies; ++i) {
        uint32_t count = tracker.countTracker[i].load_relaxed();

        uint32_t total_health = tracker.healthTracker[i].load_relaxed();

        float health_avg = (float)total_health / (float)count;
        if (count == 0) {
            health_avg = 0;
        }

        counts.counts[i] = count;
        
        // (100 is the starting health of all agents)
        // Do -2 so that the baseline reward is 0
        rewards.rewards[i] = (float)count / (float)ctx.data().initNumAgentsPerWorld +
                             health_avg / 100.0f - 2.f;

        tracker.countTracker[i].store_relaxed(0);
        tracker.healthTracker[i].store_relaxed(0);


        if (count < init_num_agents_per_species) {
            for (int e_i = count; e_i < init_num_agents_per_species; ++e_i) {
                float x_pos = ctx.data().rng.sampleUniform() * world_lim_x;
                float y_pos = ctx.data().rng.sampleUniform() * world_lim_y;

                auto entity = makeAgent(
                        ctx,
                        ma::math::Vector3{ x_pos, y_pos, 1.f },
                        i + 1,
                        100);
            }
        }
    }
}

inline void rewardSystem(Engine &ctx,
                         ma::base::Position &position,
                         Species &species,
                         Health &health,
                         AgentObservationBridge &bridge,
                         AgentStats &agent_stats)
{
    SpeciesReward &species_rew = ctx.get<SpeciesReward>(
            ctx.data().speciesInfoTracker);

    Reward &reward = ctx.get<Reward>(bridge.obsEntity);

    reward.v = species_rew.rewards[species.speciesID] +
               health.v / 100.f - 0.5f;


    // Agent gets penalized for being at the ends of the world
    float world_lim_x = ctx.data().numChunksX * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;
    float world_lim_y = ctx.data().numChunksY * ChunkInfo::kChunkWidth * 
                        ctx.data().cellDim;

    float penalty_radius = 4.f;
    if (position.x < penalty_radius || position.y < penalty_radius ||
        position.x > world_lim_x - penalty_radius ||
        position.y > world_lim_y - penalty_radius) {
        reward.v -= 1.f;
    }



    if (agent_stats.reproduced) {
        reward.v += 10.f;
    }

    if (agent_stats.hitFriendlyAgent) {
        reward.v -= 5.f;
    }

    if (agent_stats.hitEnemyAgent) {
        reward.v += 15.f;
    }

    if (agent_stats.ateFood) {
        reward.v += 7.f;
    }

    agent_stats.reproduced = 0;
    agent_stats.hitFriendlyAgent = 0;
    agent_stats.hitEnemyAgent = 0;
    agent_stats.ateFood = 0;
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

// Copies the "normal" observations (we still need to copy the sensor obs).
inline void shiftObservationsSystem(
        Engine &ctx,
        const SpeciesObservation &species_obs,
        const PositionObservation &pos_obs,
        const HealthObservation &health_obs,
        const SurroundingObservation &sur_obs,
        const Reward &rew_obs,
        const Action &act_obs,
        const StatsObservation &stats_obs,
        PrevSpeciesObservation &prev_species_obs,
        PrevPositionObservation &prev_pos_obs,
        PrevHealthObservation &prev_health_obs,
        PrevSurroundingObservation &prev_sur_obs,
        PrevReward &prev_rew_obs,
        PrevAction &prev_act_obs,
        PrevStatsObservation &prev_stats_obs)
{
    prev_species_obs.speciesID = species_obs.speciesID;
    prev_pos_obs.pos = pos_obs.pos;
    prev_health_obs.v = health_obs.v;
    prev_sur_obs.presenceHeuristic = sur_obs.presenceHeuristic;
    prev_sur_obs.movementHeuristic = sur_obs.movementHeuristic;
    prev_rew_obs.v = rew_obs.v;

    prev_act_obs.forward = act_obs.forward;
    prev_act_obs.backward = act_obs.backward;
    prev_act_obs.rotateLeft = act_obs.rotateLeft;
    prev_act_obs.rotateRight = act_obs.rotateRight;
    prev_act_obs.shoot = act_obs.shoot;
    prev_act_obs.breed = act_obs.breed;

    prev_stats_obs.hitFriendlyAgent = stats_obs.hitFriendlyAgent;
    prev_stats_obs.hitEnemyAgent = stats_obs.hitFriendlyAgent;
    prev_stats_obs.ateFood = stats_obs.ateFood;
    prev_stats_obs.reproduced = stats_obs.reproduced;
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
            ma::render::RenderCamera,
            AgentObservationBridge,
            Species,
            AgentStats
        >>({add_food_sys});

    auto health_sync_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        healthSync,
            ma::Entity,
            ma::base::Position,
            Health,
            HealthAccumulator,
            Action,
            Species,
            ma::render::RenderCamera,
            AgentStats,
        >>({action_sys});

    auto update_surrounding_obs_sys = builder.addToGraph<ma::ParallelForNode<Engine,
         updateSurroundingObservation,
            ma::Entity,
            ma::base::Position,
            AgentType,
            SurroundingObservation
         >>({health_sync_sys});

    // Required
    auto clear_tmp = builder.addToGraph<ma::ResetTmpAllocNode>(
            {update_surrounding_obs_sys});

    auto update_species_tracker = builder.addToGraph<ma::ParallelForNode<Engine,
        speciesTrackerUpdate,
            ma::Entity,
            AgentObservationBridge,
            Species,
            Health
        >>({clear_tmp});

    auto species_info_sys = builder.addToGraph<ma::ParallelForNode<Engine,
         speciesInfoSync,
            SpeciesInfoTracker,
            SpeciesCount,
            SpeciesReward
        >>({update_species_tracker});

    auto sort_agents = queueSortByWorld<Agent>(
        builder, {species_info_sys});

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
            SurroundingObservation,
            AgentStats,
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
            Health,
        >>({sort_obs});

    // Conditionally reset the world if the episode is over
    auto reward_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        rewardSystem,
            ma::base::Position,
            Species,
            Health,
            AgentObservationBridge,
            AgentStats
        >>({update_sensor_idx});

    auto sort_species = queueSortByWorld<SpeciesInfoArchetype>(
            builder, {reward_sys});

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

static void setupShiftObservationsTasks(ma::TaskGraphBuilder &builder,
                                        const Sim::Config &cfg)
{
    auto shift_obs_sys = builder.addToGraph<ma::ParallelForNode<Engine,
        shiftObservationsSystem,
            // Basically just loop through all components of the observation
            // archetype
            SpeciesObservation,
            PositionObservation,
            HealthObservation,
            SurroundingObservation,
            Reward,
            Action,
            StatsObservation,
            PrevSpeciesObservation,
            PrevPositionObservation,
            PrevHealthObservation,
            PrevSurroundingObservation,
            PrevReward,
            PrevAction,
            PrevStatsObservation
        >>({});

    (void)shift_obs_sys;
}

// Build the task graph
void Sim::setupTasks(ma::TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
    setupSensorTasks(taskgraph_mgr.init(TaskGraphID::Sensor), cfg);
    setupShiftObservationsTasks(
            taskgraph_mgr.init(TaskGraphID::ShiftObservations), cfg);
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
      currentNumFood(0),
      initNumAgentsPerWorld(cfg.initNumAgentsPerWorld)
{
    initWorld(ctx, numChunksX, numChunksY, cfg.initNumAgentsPerWorld);

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
