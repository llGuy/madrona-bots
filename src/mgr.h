#pragma once

#include <memory>

struct Manager {
    // Configures the simulation
    struct Config {
        // ...
    };



    Manager(const Config &cfg);
    ~Manager();

    void step();

private:
    struct Impl;

    std::unique_ptr<Impl> mImpl;
};
