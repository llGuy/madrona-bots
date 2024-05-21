#include <memory>

#include <madrona/py/utils.hpp>

class Manager {
public:
    struct Config {
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
