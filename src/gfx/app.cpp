#include "gfx/gfx.hpp"

namespace ma = madrona;

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    ScriptBotsViewer::Config cfg = {
        .gpuID = 0,
        .numWorlds = 4,
        .randSeed = 0,
        .initNumAgentsPerWorld = 16,
        .windowWidth = 1385,
        .windowHeight = 768
    };

    ScriptBotsViewer viewer(cfg);

    Manager *mgr = viewer.getManager();

    viewer.loop([&]() { mgr->step(); });
}
