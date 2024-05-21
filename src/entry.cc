#include <stdio.h>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mgr.h"

namespace nb = nanobind;
using namespace nb::literals;

void greet()
{
    // Manager mgr;
    // mgr.greet();
}

NB_MODULE(madrona_bots, m) {
    m.def("greet", &greet, "Greet the user!");
}
