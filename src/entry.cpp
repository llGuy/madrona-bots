#include "mgr.hpp"

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
                            int64_t rand_seed) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"))
        .def("step", &Manager::step)
#if 0
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
        .def("room_entity_observations_tensor",
             &Manager::roomEntityObservationsTensor)
        .def("door_observation_tensor",
             &Manager::doorObservationTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
#endif
    ;
}
