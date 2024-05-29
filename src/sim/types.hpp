#pragma once

#include <madrona/sync.hpp>
#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

namespace ma = madrona;

namespace mbots {

struct WorldReset {
    int32_t reset;
};

struct Action {
    // Whatever the actions may be
    int32_t forward;
    int32_t backward;
    int32_t rotate;
    int32_t shoot;
};

struct Reward {
    float v;
};

struct Done {
    int32_t v;
};

enum class AgentType {
    Herbivore,
    Carnivore,
    NumAgentTypes
};

// Each center of the chunk stores some amount of information about how many
// entities are within that chunk, what they velocities are, etc...
//
// This allows us to create a heuristic for the amount of activity there is
// near any given point.
struct ChunkInfo {
    // Each chunk is 16 cells wide.
    static constexpr uint32_t kChunkWidth = 16;

    ma::AtomicU32 numAgents;

    // This isn't exactly speed, but some heuristic for speed
    ma::AtomicU32 totalSpeed;

    // This entity points to a ChunkData archetype. We create this indirection
    // for potential memory saving. It also makes it so that we don't have to
    // sort the bigger `ChunkData` struct around (because we are going to be
    // sorting the chunk infos by world once).
    ma::Entity chunkDataEntity;

    ma::math::Vector2 chunkCoord;

    // Contains food data (or whatever other data we might need to store).
    uint8_t data[kChunkWidth * kChunkWidth];
};

struct ChunkInfoArchetype : ma::Archetype<
    ChunkInfo
> {};

struct SurroundingObservation {
    // Heuristic for measuring how many agents are around me
    float presenceHeuristic;

    // Heuristic for measuring how much movement is happening around me
    float movementHeuristic;
};

struct Health {
    int32_t v;
};

struct HealthObservation : Health {
    // Nothing
};

// We need to separate out the Health and HealthAccumulator for concurrency
// reasons.
struct HealthAccumulator {
    ma::AtomicI32 v;
};

struct BridgeSync {
    // This doesn't have anything
};

struct AddFoodSingleton {
    // This doesn't have anything
};

struct PositionObservation {
    ma::math::Vector2 pos;
};

struct Species {
    uint32_t speciesID;
};

struct SpeciesObservation : Species {
    // This is just inherited
};

struct AgentObservationBridge {
    ma::Entity obsEntity;
};

struct SensorOutputIndex {
    uint32_t idx;
};

struct Agent : ma::Archetype<
    ma::base::Position,
    ma::base::Rotation,
    ma::base::Scale,
    ma::base::ObjectID,


    // Properties
    Species,
    AgentType,
    Health,
    HealthAccumulator,

 
    // Input
    Action,
 

    // TODO: Observations
    SurroundingObservation,

 
    // Reward, episode termination
    Reward,
    Done,


    // Required or sensory input.
    ma::render::Renderable,
    ma::render::RenderCamera,

    AgentObservationBridge,

    // Used by the visualizer to get the correct raycast output.
    SensorOutputIndex
> {};

// This is an archetype just for observations.
//
// This is because the observations have to be sorted by species.
struct AgentObservationArchetype : ma::Archetype<
    SpeciesObservation,
    PositionObservation,
    HealthObservation,
    SurroundingObservation
> {};

// This is mostly for the visualizer
struct StaticObject : ma::Archetype<
    ma::base::Position,
    ma::base::Rotation,
    ma::base::Scale,
    ma::base::ObjectID,
    ma::render::Renderable
> {};

}
