namespace mbots {

template <typename ArchetypeT>
ma::Entity Engine::makeRenderableEntity()
{
    ma::Entity e = makeEntity<ArchetypeT>();
    ma::render::RenderingSystem::makeEntityRenderable(*this, e);
    
    return e;
}

inline void Engine::destroyRenderableEntity(ma::Entity e)
{
    ma::render::RenderingSystem::cleanupRenderableEntity(*this, e);
    destroyEntity(e);
}

inline ma::Entity Engine::makeAgent()
{
    ma::Entity e = makeRenderableEntity<Agent>();

    // Create the observation entity too
    get<AgentObservationBridge>(e).obsEntity =
        makeEntity<AgentObservationArchetype>();

    return e;
}

inline void Engine::destroyAgent(ma::Entity e)
{
    // Destroy the observation entity
    ma::Entity bridge_e = get<AgentObservationBridge>(e).obsEntity;

    destroyEntity(bridge_e);

    destroyRenderableEntity(e);
}

inline ma::math::Vector2 Sim::getChunkCoord(
        const ma::math::Vector2 &world_pos)
{
    // Turn the world space position into cell position
    ma::math::Vector2 cell_pos = world_pos / cellDim;
    ma::math::Vector2 chunk_coord = cell_pos / (float)ChunkInfo::kChunkWidth;
    chunk_coord.x = std::floor(chunk_coord.x);
    chunk_coord.y = std::floor(chunk_coord.y);
    return chunk_coord;
}

inline int32_t Sim::getChunkIndex(
        const ma::math::Vector2 &chunk_coord)
{
    int32_t x = (int32_t)chunk_coord.x;
    int32_t y = (int32_t)chunk_coord.y;

    if (x < 0 || y < 0 || 
        x >= (int32_t)numChunksX || 
        y >= (int32_t)numChunksY) {
        return -1;
    } else {
        return x + y * numChunksX;
    }
}

inline ChunkInfo *Sim::getChunkInfo(Engine &ctx, int32_t chunk_idx)
{
    if (chunk_idx == -1) {
        return nullptr;
    }

    ma::Loc loc = chunksLoc;
    loc.row += chunk_idx;
    return &ctx.get<ChunkInfo>(loc);
}

inline bool FoodPackage::consume(Engine &ctx)
{
    uint32_t prev_value = numFood.load_relaxed();

    if (prev_value == 0)
        return false;

    bool success = numFood.compare_exchange_weak<
            ma::sync::memory_order::relaxed,
            ma::sync::memory_order::relaxed
        >(prev_value, prev_value - 1);

    if (success) {
        if (prev_value - 1 == 0) {
            // Now, there is no more food here!
            ctx.destroyRenderableEntity(foodEntity);
            ctx.data().currentNumFood.fetch_add_relaxed(-1);
        }

        return true;
    } else {
        return false;
    }
}

}
