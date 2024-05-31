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
                            int64_t rand_seed,
                            int64_t sensor_size) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .randSeed = (uint32_t)rand_seed,
                .sensorSize = (uint32_t)sensor_size
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("rand_seed"),
           nb::arg("sensor_size"))
        .def("step", &Manager::step)
        .def("sensor_tensor", &Manager::sensorTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("position_tensor", &Manager::positionTensor)
        .def("health_tensor", &Manager::healthTensor)
        .def("surrounding_tensor", &Manager::surroundingTensor)
    ;
}
