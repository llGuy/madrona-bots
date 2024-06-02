#pragma once

#include <madrona/sync.hpp>
#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

namespace ma = madrona;

namespace mbots {

inline constexpr uint32_t kNumSpecies = 4;

class Engine;

struct WorldReset {
    int32_t reset;
};

struct Action {
    // Whatever the actions may be
    int32_t forward;
    int32_t backward;
    int32_t rotateLeft;
    int32_t rotateRight;
    int32_t shoot;

    // Breeding can only happen if the agent has more than 60 health because
    // breeding will cause one of the agents to lose 40 health.
    int32_t breed;
};

struct PrevAction : Action {};

struct Reward {
    float v;
};

struct PrevReward : Reward {};

struct Done {
    int32_t v;
};

enum class AgentType {
    Herbivore,
    Carnivore,
    NumAgentTypes
};

struct FoodPackage {
    ma::AtomicU32 numFood;

    // For visualization purposes.
    ma::Entity foodEntity;

    uint8_t x, y;

    // Returns whether consumption was successful
    inline bool consume(Engine &ctx);
};

// Each center of the chunk stores some amount of information about how many
// entities are within that chunk, what they velocities are, etc...
//
// This allows us to create a heuristic for the amount of activity there is
// near any given point.
struct ChunkInfo {
    // Each chunk is 16 cells wide.
    static constexpr uint32_t kChunkWidth = 16;
    static constexpr uint32_t kMaxFoodPackages = 5;
    static constexpr uint32_t kMaxFoodPerPackage = 1;

    ma::AtomicU32 numAgents;

    // This isn't exactly speed, but some heuristic for speed
    ma::AtomicU32 totalSpeed;

    // This entity points to a ChunkData archetype. We create this indirection
    // for potential memory saving. It also makes it so that we don't have to
    // sort the bigger `ChunkData` struct around (because we are going to be
    // sorting the chunk infos by world once).
    ma::Entity chunkDataEntity;

    ma::math::Vector2 chunkCoord;

    // We are going to change the way food is handled.
    // Each chunk can have at most `kMaxFoodPackages` packages of food.
    FoodPackage foodPackages[kMaxFoodPackages];

#if 0
    // Contains food data (or whatever other data we might need to store).
    uint8_t data[kChunkWidth * kChunkWidth];
#endif
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

struct PrevSurroundingObservation : SurroundingObservation {};

struct Health {
    int32_t v;
};

struct HealthObservation : Health {};
struct PrevHealthObservation : HealthObservation {};

// We need to separate out the Health and HealthAccumulator for concurrency
// reasons.
struct HealthAccumulator {
    ma::AtomicI32 v;
};

struct BridgeSync {};
struct AddFoodSingleton {};

struct PositionObservation {
    ma::math::Vector2 pos;
};

struct PrevPositionObservation : PositionObservation {};

struct Species {
    uint32_t speciesID;
};

struct SpeciesObservation : Species {};
struct PrevSpeciesObservation : SpeciesObservation {};

struct AgentObservationBridge {
    ma::Entity obsEntity;
};

struct SensorOutputIndex {
    uint32_t idx;
};

struct SpeciesInfoTracker {
    ma::AtomicU32 countTracker[kNumSpecies];
    ma::AtomicU32 healthTracker[kNumSpecies];
};

struct SpeciesCount {
    uint32_t counts[kNumSpecies];
};

struct SpeciesReward {
    float rewards[kNumSpecies];
};

// We just need to create one of these per world
struct SpeciesInfoArchetype : ma::Archetype<
    SpeciesInfoTracker,
    SpeciesCount,
    SpeciesReward
> {};

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
    SurroundingObservation,
    Reward,

    // The agent observation archetype has the action too because
    // of ordering (sorting based on species index)
    Action,



    // Now, store all the previous actions for training purposes.
    PrevSpeciesObservation,
    PrevPositionObservation,
    PrevHealthObservation,
    PrevSurroundingObservation,
    PrevReward,
    PrevAction
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
