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
    ma::AtomicU32 numAgents;
    ma::AtomicFloat totalSpeed;

    // This entity points to a ChunkData archetype. We create this indirection
    // for potential memory saving. It also makes it so that we don't have to
    // sort the bigger `ChunkData` struct around (because we are going to be
    // sorting the chunk infos by world once).
    ma::Entity chunkDataEntity;
};

struct ChunkInfoArchetype : ma::Archetype<
    ChunkInfo
> {};

struct ChunkData {
    // Each chunk is 16 cells wide.
    static constexpr uint32_t kChunkWidth = 16;

    // Contains food data (or whatever other data we might need to store).
    uint8_t data[kChunkWidth * kChunkWidth];
};

struct ChunkDataArchetype : ma::Archetype<
    ChunkData
> {};

struct Agent : ma::Archetype<
    ma::base::Position,
    ma::base::Rotation,
    ma::base::Scale,
    ma::base::ObjectID,


    // Properties
    AgentType,

 
    // Input
    Action,
 

    // TODO: Observations

 
    // Reward, episode termination
    Reward,
    Done,


    // Required or sensory input.
    ma::render::Renderable,
    ma::render::RenderCamera
> {};

}
