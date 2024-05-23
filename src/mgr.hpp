#include <memory>

#include <madrona/py/utils.hpp>

namespace ma = madrona;

class Manager {
public:
    struct Config {
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
        uint32_t sensorSize; // Number of pixels in the sensor
        uint32_t numAgentsPerWorld;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    ma::py::Tensor sensorTensor() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
