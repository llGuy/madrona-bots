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
        
        // If we are doing visualization
        void *renderBridge = nullptr;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    ma::py::Tensor sensorTensor() const;

    void setAction(uint32_t agent_idx,
                   int32_t forward,
                   int32_t backward,
                   int32_t rotate,
                   int32_t shoot);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
