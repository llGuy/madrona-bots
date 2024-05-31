#include "mgr.hpp"
#include <stdio.h>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    Manager::Config cfg = {
        .gpuID = 0,
        .numWorlds = 1,
        .randSeed = 0,
        .sensorSize = 32,
        .numAgentsPerWorld = 2,
    };

    Manager mgr(cfg);



    // Just visualizes the first world's agents' sensor
    [[maybe_unused]] auto viz_sensor = [&mgr, &cfg] (uint32_t agent_idx) {
        int64_t num_bytes = cfg.sensorSize;
        uint8_t *print_ptr = (uint8_t *)ma::cu::allocReadback(num_bytes);

        uint8_t *sensor_tensor = (uint8_t *)(mgr.depthTensor().devicePtr());

        cudaMemcpy(print_ptr, sensor_tensor + agent_idx * num_bytes,
                num_bytes,
                cudaMemcpyDeviceToHost);

        for (int i = 0; i < (int)cfg.sensorSize; ++i) {
            printf("%u  ", print_ptr[i]);
        }

        printf("\n");
    };

    char action_desc_buffer[256];

    while (true) {
        if (!fgets(action_desc_buffer, sizeof(action_desc_buffer), stdin)) {
            // There was a problem
            break;
        }

        for (int i = 0; i < (int)strlen(action_desc_buffer)-1; ++i) {
            char act = action_desc_buffer[i];

            int32_t forward = 0, backward = 0, rotate = 0, shoot = 0;

            switch (act) {
            case 'f': {
                forward = 1;
            } break;

            case 'b': {
                backward = 1;
            } break;

            case 'r': {
                rotate = 1;
            } break;

            case 's': {
                shoot = 1;
            } break;

            default: {
            } break;
            }

            mgr.setAction(0, forward, backward, rotate, shoot);

            mgr.step();

            viz_sensor(0);
        }

        memset(action_desc_buffer, 0, sizeof(action_desc_buffer));
    }

    return 0;
}
