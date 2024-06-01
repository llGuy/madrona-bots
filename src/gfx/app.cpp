#include "entry/mgr.hpp"

#include <imgui.h>

#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>

namespace ma = madrona;

// Random color table
static constexpr ImU32 kRandomColorTable[] = {
    IM_COL32(0, 0, 0, 255),
    IM_COL32(170, 0, 0, 255),
    IM_COL32(0, 170, 0, 255),
    IM_COL32(0, 0, 170, 255),
    IM_COL32(170, 0, 170, 255),
    IM_COL32(0, 170, 170, 255),
    IM_COL32(170, 170, 0, 255),
    IM_COL32(170, 170, 170, 255),
    IM_COL32(85, 85, 85, 255),
    IM_COL32(255, 85, 85, 255),
    IM_COL32(85, 255, 85, 255),
    IM_COL32(85, 85, 255, 255),
    IM_COL32(85, 255, 255, 255),
    IM_COL32(255, 255, 85, 255),
    IM_COL32(255, 85, 255, 255),
    IM_COL32(255, 255, 255, 255)
};

static constexpr uint32_t kNumRandomColors = sizeof(kRandomColorTable) /
    sizeof(kRandomColorTable[0]);

// In assets.cpp
void loadRenderObjects(ma::render::RenderManager &render_mgr);

// Most of this code is setup for the visualization renderer.
static inline ma::render::RenderManager initRenderManager(
    ma::render::APIBackend *backend,
    ma::render::GPUDevice *gpu_device)
{
    return ma::render::RenderManager(backend, gpu_device, {
        .enableBatchRenderer = false,

        // This is irrelevant because we aren't using the batch rasterizer
        .agentViewWidth = 32,
        .agentViewHeight = 32,

        // We are only going to visualize one world.
        .numWorlds = 1,

        .maxViewsPerWorld = 2048,
        .maxInstancesPerWorld = 2048,
        .execMode = ma::ExecMode::CUDA,
    });
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    ma::WindowManager wm {};
    ma::WindowHandle window = wm.makeWindow(
            "Script Bots Visualizer", 1365, 768);

    ma::render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // The RenderManager is the thing which is actually responsible for 
    // rendering the visualization to the screen.
    ma::render::RenderManager render_mgr = initRenderManager(
        wm.gpuAPIManager().backend(),
        render_gpu.device());
    *render_mgr.exportedWorldID() = 0;

    loadRenderObjects(render_mgr);

    Manager::Config cfg = {
        .gpuID = 0,
        .numWorlds = 3,
        .randSeed = 0,
        .sensorSize = 32,
        .renderBridge = (void *)render_mgr.bridge()
    };

    Manager mgr(cfg);

    ma::math::Quat initial_camera_rotation =
        (ma::math::Quat::angleAxis(-ma::math::pi / 2.f, ma::math::up) *
        ma::math::Quat::angleAxis(-ma::math::pi / 2.f, ma::math::right)).normalize();

    ma::viz::Viewer viewer(render_mgr, window.get(), {
        .numWorlds = cfg.numWorlds,
        .simTickRate = 25u,
        .cameraMoveSpeed = 10.f,
        .cameraPosition = { 0.f, 0.f, 40 },
        .cameraRotation = initial_camera_rotation,
    });

    uint32_t inspecting_agent_idx = 0;
    uint32_t inspecting_world_idx = 0;

    // Readback for the sensor information
    int64_t num_bytes = cfg.sensorSize;
    uint8_t *depth_ptr = (uint8_t *)ma::cu::allocReadback(num_bytes);
    uint8_t *semantic_ptr = (uint8_t *)ma::cu::allocReadback(num_bytes);

    uint32_t *sensor_idx_ptr = (uint32_t *)ma::cu::allocReadback(sizeof(uint32_t));

    viewer.loop(
        // Function for controling input that affects the whole world
        [&](ma::CountT world_idx, 
                              const ma::viz::Viewer::UserInput &input) {
            using Key = ma::viz::Viewer::KeyboardKey;
            (void)input;
        }, 
        // Function for controlling input that affects an agent
        [&](ma::CountT world_idx, ma::CountT agent_idx,
                              const ma::viz::Viewer::UserInput &input) {
            using Key = ma::viz::Viewer::KeyboardKey;

            int32_t forward = 0, backward = 0, 
                    rotate_left = 0, rotate_right = 0,
                    shoot = 0, breed = 0;

            if (input.keyPressed(Key::W)) forward = 1;
            if (input.keyPressed(Key::S)) backward = 1;
            if (input.keyPressed(Key::R)) rotate_left = 1;
            if (input.keyPressed(Key::F)) rotate_right = 1;
            if (input.keyPressed(Key::Space)) shoot = 1;
            if (input.keyPressed(Key::Q)) breed = 1;

            // For now, we only control the agent of the first world.
            mgr.setAction(agent_idx + mgr.agentOffsetForWorld(world_idx), 
                          forward, backward, rotate_left, rotate_right, shoot, breed);

            inspecting_agent_idx = agent_idx;
            inspecting_world_idx = world_idx;
        }, 
        // Function for controlling what happens during a step
        [&]() {
            mgr.step();
            render_mgr.readECS();
        }, 
        // Function for controlling extra UI we might want to have.
        [&]() {
            // Does nothing for now
            ImGui::Begin("Raycast Visualizer");

            int vert_off = 45;
            float pix_scale = 20;

            uint32_t rt_output_offset = 0;

            uint8_t *depth_tensor = (uint8_t *)(mgr.depthTensor().devicePtr());
            int8_t *semantic_tensor = (int8_t *)(mgr.semanticTensor().devicePtr());

            uint32_t *sensor_idx_tensor = (uint32_t *)(mgr.sensorIndexTensor().devicePtr());

            int32_t global_agent_idx = mgr.agentOffsetForWorld(inspecting_world_idx) +
                                        inspecting_agent_idx;

            cudaMemcpy(sensor_idx_ptr, sensor_idx_tensor + global_agent_idx,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(depth_ptr, depth_tensor + (*sensor_idx_ptr) * num_bytes,
                    num_bytes,
                    cudaMemcpyDeviceToHost);

            cudaMemcpy(semantic_ptr, semantic_tensor + (*sensor_idx_ptr) * num_bytes,
                    num_bytes,
                    cudaMemcpyDeviceToHost);

            uint32_t num_forward_rays = 3 * 32 / 4;
            uint32_t num_backward_rays = 1 * 32 / 4;

            auto draw_list = ImGui::GetWindowDrawList();
            ImVec2 window_pos = ImGui::GetWindowPos();

            // Depth information
            for (int i = 0; i < num_forward_rays; ++i) {
                auto realColor = IM_COL32(
                        (uint8_t)depth_ptr[i],
                        (uint8_t)depth_ptr[i],
                        (uint8_t)depth_ptr[i],
                        255);

                draw_list->AddRectFilled(
                        { ((i) * pix_scale) + window_pos.x, 
                        ((0) * pix_scale) + window_pos.y + vert_off }, 
                        { ((i + 1) * pix_scale) + window_pos.x,   
                        ((1) * pix_scale) + +window_pos.y + vert_off },
                        realColor, 0, 0);
            }

            for (int i = 0; i < num_backward_rays; ++i) {
                auto realColor = IM_COL32(
                        (uint8_t)depth_ptr[num_forward_rays + i],
                        (uint8_t)depth_ptr[num_forward_rays + i],
                        (uint8_t)depth_ptr[num_forward_rays + i],
                        255);

                draw_list->AddRectFilled(
                        { ((i) * pix_scale) + window_pos.x, 
                        ((0) * pix_scale) + window_pos.y + pix_scale + vert_off }, 
                        { ((i + 1) * pix_scale) + window_pos.x,   
                        ((1) * pix_scale) + +window_pos.y + pix_scale + vert_off },
                        realColor, 0, 0);
            }

            vert_off += pix_scale * 2;

            // Semantic information
            for (int i = 0; i < num_forward_rays; ++i) {
                int8_t semantic_info = semantic_ptr[i] + 1;

                auto realColor = kRandomColorTable[semantic_info % kNumRandomColors];

                draw_list->AddRectFilled(
                        { ((i) * pix_scale) + window_pos.x, 
                        ((0) * pix_scale) + window_pos.y + vert_off }, 
                        { ((i + 1) * pix_scale) + window_pos.x,   
                        ((1) * pix_scale) + +window_pos.y + vert_off },
                        realColor, 0, 0);
            }

            for (int i = 0; i < num_backward_rays; ++i) {
                int8_t semantic_info = semantic_ptr[num_forward_rays + i] + 1;

                auto realColor = kRandomColorTable[semantic_info % kNumRandomColors];

                draw_list->AddRectFilled(
                        { ((i) * pix_scale) + window_pos.x, 
                        ((0) * pix_scale) + window_pos.y + pix_scale + vert_off }, 
                        { ((i + 1) * pix_scale) + window_pos.x,   
                        ((1) * pix_scale) + +window_pos.y + pix_scale + vert_off },
                        realColor, 0, 0);
            }

            ImGui::End();
        });
}
