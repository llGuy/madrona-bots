#include "mgr.hpp"
#include <stdio.h>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    Manager::Config cfg = {
        .gpuID = 0,
        .numWorlds = 2,
        .randSeed = 0,
        .sensorSize = 32,
        .numAgentsPerWorld = 2,
    };

    Manager mgr(cfg);



    // Just visualizes the first world's agents' sensor
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

    for (int i = 0; i < 4; ++i) {
        mgr.step();

        // viz_sensor(0);
    }

    return 0;
}
