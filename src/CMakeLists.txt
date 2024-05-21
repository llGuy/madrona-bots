set(SIMULATOR_SRCS
    sim.h
    sim.cc
)

add_library(madrona_bots_mgr STATIC
    mgr.cc mgr.h
)

target_link_libraries(madrona_bots_mgr
    PUBLIC
        madrona_python_utils
    PRIVATE
        madrona_common
)

madrona_python_module(madrona_bots
    entry.cc
)

SET(MADRONA_BOTS_COMPILE_FLAGS
    -v)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madrona_bots_gpu_srcs
        SOURCES_DEFN
            MADRONA_BOTS_EMPTY
        FLAGS_DEFN
            MADRONA_BOTS_COMPILE_FLAGS
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_bots_mgr PRIVATE
        madrona_bots_gpu_srcs
        madrona_mw_gpu
    )
else()
    message(FATAL_ERROR "GPU code not supported!")
endif()


