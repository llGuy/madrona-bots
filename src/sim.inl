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

}
