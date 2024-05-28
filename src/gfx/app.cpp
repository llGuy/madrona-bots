#include "entry/mgr.hpp"

#include <imgui.h>

#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>

namespace ma = madrona;

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

    loadRenderObjects(render_mgr);

    Manager::Config cfg = {
        .gpuID = 0,
        .numWorlds = 1,
        .randSeed = 0,
        .sensorSize = 32,
        .numAgentsPerWorld = 2,
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

    [[maybe_unused]] auto viz_sensor = [&mgr, &cfg] (uint32_t agent_idx) {
        int64_t num_bytes = cfg.sensorSize;
        uint8_t *print_ptr = (uint8_t *)ma::cu::allocReadback(num_bytes);

        uint8_t *sensor_tensor = (uint8_t *)(mgr.sensorTensor().devicePtr());

        cudaMemcpy(print_ptr, sensor_tensor + agent_idx * num_bytes,
                num_bytes,
                cudaMemcpyDeviceToHost);

        for (int i = 0; i < (int)cfg.sensorSize; ++i) {
            printf("%u  ", print_ptr[i]);
        }

        printf("\n");
    };

    auto world_input_fn = [&](ma::CountT world_idx, 
                              const ma::viz::Viewer::UserInput &input) {
        using Key = ma::viz::Viewer::KeyboardKey;

        // Any sort of world-wide control like resetting or something.
    };

    auto agent_input_fn = [&](ma::CountT world_idx, ma::CountT agent_idx,
                              const ma::viz::Viewer::UserInput &input) {
        using Key = ma::viz::Viewer::KeyboardKey;

        int32_t forward = 0, backward = 0, rotate = 0, shoot = 0;

        if (input.keyPressed(Key::W)) forward = 1;
        if (input.keyPressed(Key::S)) backward = 1;
        if (input.keyPressed(Key::R)) rotate = 1;
        if (input.keyPressed(Key::Space)) shoot = 1;

        // For now, we only control the agent of the first world.
        mgr.setAction(agent_idx, forward, backward, rotate, shoot);
    };

    auto step_fn = [&]() {
        mgr.step();

        // After the step, we need the renderer to read from the ECS.
        render_mgr.readECS();
    };

    auto ui_fn = [&]() {
        // If we want some extra control of the UI that pops up.
    };

    uint32_t inspecting_agent_idx = 0;

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

            int32_t forward = 0, backward = 0, rotate = 0, shoot = 0;

            if (input.keyPressed(Key::W)) forward = 1;
            if (input.keyPressed(Key::S)) backward = 1;
            if (input.keyPressed(Key::R)) rotate = 1;
            if (input.keyPressed(Key::Space)) shoot = 1;

            // For now, we only control the agent of the first world.
            mgr.setAction(agent_idx, forward, backward, rotate, shoot);

            // viz_sensor(agent_idx);

            inspecting_agent_idx = agent_idx;
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

            int64_t num_bytes = cfg.sensorSize;
            uint8_t *print_ptr = (uint8_t *)ma::cu::allocReadback(num_bytes);

            uint8_t *sensor_tensor = (uint8_t *)(mgr.sensorTensor().devicePtr());

            cudaMemcpy(print_ptr, sensor_tensor + inspecting_agent_idx * num_bytes,
                    num_bytes,
                    cudaMemcpyDeviceToHost);

            uint32_t num_forward_rays = 3 * 32 / 4;
            uint32_t num_backward_rays = 1 * 32 / 4;

            ImGui::Text("Raycast output at index %d\n", (int)print_ptr[0]);

            auto draw_list = ImGui::GetWindowDrawList();
            ImVec2 window_pos = ImGui::GetWindowPos();

            for (int i = 0; i < num_forward_rays; ++i) {
                auto realColor = IM_COL32(
                        (uint8_t)print_ptr[i],
                        (uint8_t)print_ptr[i],
                        (uint8_t)print_ptr[i],
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
                        (uint8_t)print_ptr[num_forward_rays + i],
                        (uint8_t)print_ptr[num_forward_rays + i],
                        (uint8_t)print_ptr[num_forward_rays + i],
                        255);

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
