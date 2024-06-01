include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SIMULATOR_SRCS
    sim/sim.hpp
    sim/sim.cpp
)

add_library(madrona_bots_mgr STATIC
    entry/mgr.cpp entry/mgr.hpp
)

target_link_libraries(madrona_bots_mgr
    PUBLIC
        madrona_python_utils
    PRIVATE
        madrona_common
)

SET(MADRONA_BOTS_COMPILE_FLAGS
    -v)

if (TARGET madrona_mw_gpu)
    message(INFO ${SIMULATOR_SRCS})

    madrona_build_compile_defns(
        OUT_TARGET
            madrona_bots_gpu_srcs
        SOURCES_DEFN
            MBOTS_SRC_LIST
        FLAGS_DEFN
            MBOTS_COMPILE_FLAGS
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_bots_mgr PUBLIC
        madrona_bots_gpu_srcs
        madrona_mw_gpu
    )

    message(STATUS "GPU code supported!")
else()
    message(FATAL_ERROR "GPU code not supported!")
endif()

add_executable(madrona_bots_test
    entry/test.cpp
)

target_link_libraries(madrona_bots_test PRIVATE
    madrona_bots_mgr
)

madrona_python_module(madrona_bots
    entry/entry.cpp
)

target_link_libraries(madrona_bots PRIVATE
    madrona_bots_mgr
)

add_executable(madrona_bots_app
    gfx/app.cpp
    gfx/assets.cpp
    gfx/gfx.cpp
)

target_link_libraries(madrona_bots_app PRIVATE
    madrona_bots_mgr madrona_mw_core madrona_viz madrona_cuda stb
)

target_compile_definitions(madrona_bots_app PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)
