#include <filesystem>
#include <madrona/importer.hpp>
#include <madrona/render/common.hpp>
#include <madrona/render/render_mgr.hpp>

#include "sim/sim.hpp"

namespace ma = madrona;

static std::vector<ma::imp::SourceMaterial> makeMaterials()
{
    // This defines the following ordering:
    //
    // Material 0 is for gray objects (like the wall).
    // Material 1 is for for orange objects (like the food).
    // Material 2 is for the textured plane.
    // Material 3 is for the agent's body (smiley face).
    // Material 4 is for the agent's hands (white).
    return {
        { ma::math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { ma::render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { ma::math::Vector4{0.5f, 0.3f, 0.3f, 0.0f}, 0, 0.8f, 0.2f,},
        { ma::math::Vector4{0.1f, 0.1f, 1.0f, 0.0f}, 1, 0.8f, 1.0f,},
        { ma::render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
    };
}

void loadRenderObjects(ma::render::RenderManager &render_mgr)
{
    using namespace mbots;

    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;

    // The agent will be the smiley sphere guy.
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();

    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();

    // Food will just be rendered as cubes.
    render_asset_paths[(size_t)SimObject::Food] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();



    // Convert these paths to simple C-strings.
    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }


    // Perform the import
    std::array<char, 1024> import_err;
    auto render_assets = ma::imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, ma::Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    // This defines the following ordering for textures:
    //
    // Texture 0 is the green grid.
    // Texture 1 is the smiley face.
    //
    // Such that, any time a texture is referenced in the materials, the 
    // ordering is defined by the above.
    auto green_grid_str = (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string();
    auto smile_str = (std::filesystem::path(DATA_DIR) /
           "smile.png").string();
    
    std::vector<ma::imp::SourceTexture> textures = {
        ma::imp::SourceTexture(green_grid_str.c_str()),
        ma::imp::SourceTexture(smile_str.c_str())
    };

    auto materials = makeMaterials();

    { // Override the materials
        // The agent's materials
        render_assets->objects[(uint32_t)SimObject::Agent].meshes[0].materialIDX = 3;
        render_assets->objects[(uint32_t)SimObject::Agent].meshes[1].materialIDX = 3;
        render_assets->objects[(uint32_t)SimObject::Agent].meshes[2].materialIDX = 3;

        // The wall's materials
        render_assets->objects[(uint32_t)SimObject::Wall].meshes[0].materialIDX = 0;

        // The food's materials
        render_assets->objects[(uint32_t)SimObject::Food].meshes[0].materialIDX = 1;

        // The plane's materials
        render_assets->objects[(uint32_t)SimObject::Plane].meshes[0].materialIDX = 2;
    }

    // Load the render objects into the renderer itself
    render_mgr.loadObjects(
            render_assets->objects,
            materials,
            textures);

    // Configure lighting
    render_mgr.configureLighting({
        { 
            true, 
            ma::math::Vector3{1.0f, 1.0f, -0.05f}, 
            ma::math::Vector3{1.0f, 1.0f, 1.0f}
        }
    });
}
