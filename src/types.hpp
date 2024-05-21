#pragma once

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


struct Agent : ma::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    ma::base::Position,
    ma::base::Rotation,
    ma::base::Scale,
    ma::base::ObjectID,
 
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
