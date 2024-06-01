#include <memory>

#include <vector>
#include <madrona/py/utils.hpp>

namespace ma = madrona;

class Manager {
public:
    struct Config {
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
        uint32_t initNumAgentsPerWorld; // Number of agents at the 
                                        // start of the simulation

        uint32_t sensorSize = 32; // Number of pixels in the sensor
        
        // If we are doing visualization
        void *renderBridge = nullptr;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    // This returns the semantic information that the agent sees.
    ma::py::Tensor semanticTensor() const;

    // This returns the depth information that the agent sees.
    ma::py::Tensor depthTensor() const;

    // due to indirection, we need to also return the indices of these images.
    ma::py::Tensor sensorIndexTensor() const;

    // One reward per species.
    ma::py::Tensor rewardTensor() const;
    ma::py::Tensor speciesCountTensor() const;

    ma::py::Tensor positionTensor() const;
    ma::py::Tensor healthTensor() const;
    ma::py::Tensor surroundingTensor() const;

    ma::py::Tensor actionTensor() const;

    void setAction(uint32_t agent_idx,
                   int32_t forward,
                   int32_t backward,
                   int32_t rotateLeft,
                   int32_t rotateRight,
                   int32_t shoot,
                   int32_t breed);

    uint32_t agentOffsetForWorld(uint32_t world_idx);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
