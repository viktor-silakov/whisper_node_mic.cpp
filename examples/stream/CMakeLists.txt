if (WHISPER_SDL2)
    # stream
    set(TARGET stream)
    add_executable(${TARGET} stream.cpp params.cpp audio.cpp utils.cpp)

    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})
endif ()