set(SIMULATOR_SRCS
    sim.hpp
    sim.cpp
)

add_library(madrona_bots_mgr STATIC
    mgr.cpp mgr.hpp
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
    message(INFO "EMPTTTTTTTTTTTTTTYTYYYYYYYYYYYYYYY???")
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
    test.cpp
)

target_link_libraries(madrona_bots_test PRIVATE
    madrona_bots_mgr
)

madrona_python_module(madrona_bots
    entry.cpp
)

target_link_libraries(madrona_bots PRIVATE
    madrona_bots_mgr
)
