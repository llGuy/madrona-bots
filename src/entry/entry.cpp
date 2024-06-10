#include "mgr.hpp"
#include "gfx/gfx.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_bots, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            int64_t init_num_agents_per_world) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .initNumAgentsPerWorld = (uint32_t)init_num_agents_per_world,
                .sensorSize = 32u
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("init_num_agents_per_world"))
        .def("step", &Manager::step)
        .def("shift_observations", &Manager::shiftObservations)
        .def("depth_tensor", &Manager::depthTensor)
        .def("semantic_tensor", &Manager::semanticTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("species_count_tensor", &Manager::speciesCountTensor)
        .def("position_tensor", &Manager::positionTensor)
        .def("health_tensor", &Manager::healthTensor)
        .def("surrounding_tensor", &Manager::surroundingTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("stats_tensor", &Manager::statsTensor)
        .def("hidden_state_tensor", &Manager::hiddenStateTensor)
    ;

    nb::class_<ScriptBotsViewer> (m, "ScriptBotsViewer")
        .def("__init__", [](ScriptBotsViewer *self,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t rand_seed,
                            int64_t init_num_agents_per_world,
                            int64_t window_width,
                            int64_t window_height) {
            new (self) ScriptBotsViewer(ScriptBotsViewer::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .initNumAgentsPerWorld = (uint32_t)init_num_agents_per_world,
                .windowWidth = (uint32_t)window_width,
                .windowHeight = (uint32_t)window_height
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("init_num_agents_per_world"),
           nb::arg("window_width"),
           nb::arg("window_height"))
        // .def("loop", &ScriptBotsViewer::loop)
        .def("loop", [](ScriptBotsViewer *self, 
                        int num_epochs,
                        nb::callable step_fn,
                        nb::object carry) {
            uint32_t current_epoch = 1;
            self->loop([&] () { step_fn(current_epoch++, carry); });
        }, nb::arg("num_epochs"), 
           nb::arg("step_fn"),
           nb::arg("carry"))
        .def("get_sim_mgr", &ScriptBotsViewer::getManager, nb::rv_policy::reference)
    ;
}
