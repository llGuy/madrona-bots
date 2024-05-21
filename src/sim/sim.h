#pragma once

#include <madrona/math.hpp>
#include <madrona/components.hpp>

namespace ma = madrona;

namespace mbots {
    
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    // SensorObservation,
    NumExports
};

struct Action {
    int32_t moveAmount;
    int32_t rotate;
};

struct Reward {
    float value;
};

struct Done {
    int32_t isDone;
};

struct Agent : public ma::Archetype<
    // Base requirements
    ma::base::Position,
    ma::base::Rotation,
    ma::base::Scale,
    ma::base::ObjectID,

#if 0
    // For raytracing purposes
    ma::render::Renderable,
    ma::render::RenderCamera,
#endif

    // Functionality
    Action,
    Reward,
    Done
> {};




struct Config {
    uint32_t initialNumAgents;
    RandKey initRandKey;
};

struct WorldInit {
    // Empty for now
};

struct Sim : public ma::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);
};

// Declare the simulation definition
class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;

public:
    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}
