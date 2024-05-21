#include "mgr.hpp"

int main(int argc, char **argv)
{
    Manager::Config cfg = {
        .gpuID = 0,
        .numWorlds = 1,
        .randSeed = 0
    };

    Manager mgr(cfg);

    mgr.step();

    return 0;
}
