#pragma once

#include "types.hpp"

#include <cmath>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace ma = madrona;

namespace mbots {

class Engine;
    
// Tensors that we want to export to PyTorch
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    Sensor,
    NumExports
};

enum class TaskGraphID : uint32_t {
    Step,
    Sensor,
    NumTaskGraphs,
};

enum class SimObject : uint32_t {
    Agent,
    Wall,
    Food,
    Plane,
    NumObjects
};

struct Sim : ma::WorldBase {
    // Configuration struct for the simulation
    struct Config {
        uint32_t numAgentsPerWorld;
        ma::RandKey initRandKey;

        uint32_t numChunksX;
        uint32_t numChunksY;

        float cellDim;

        void *renderBridge;
    };

    // Per-world configuration - not needed for now.
    struct WorldInit 
    {
    };



    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(ma::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graphs
    static void setupTasks(ma::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);

    inline ma::math::Vector2 getChunkCoord(
            const ma::math::Vector2 &world_pos);

    inline int32_t getChunkIndex(
            const ma::math::Vector2 &chunk_coord);

    inline ChunkInfo *getChunkInfo(Engine &ctx, int32_t chunk_idx);



    uint32_t curWorldEpisode;

    // The base random key that episode random keys are split off of
    // and the random number generator
    ma::RandKey initRandKey;
    ma::RNG rng;

    // Should the environment automatically reset (generate a new episode)
    // at the end of each episode?
    bool autoReset;

    // Number of agents in this world.
    uint32_t numAgents;

    uint32_t numChunksX;
    uint32_t numChunksY;

    // How long the cell is in meters. A chunk is made up of 
    // ChunkData::kChunkWidth x ChunkData::kChunkWidth cells.
    float cellDim;

    ma::Loc chunksLoc;
};

class Engine : public ::ma::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    template <typename ArchetypeT>
    inline ma::Entity makeRenderableEntity();

    inline void destroyRenderableEntity(ma::Entity e);
};

}

#include "sim.inl"
