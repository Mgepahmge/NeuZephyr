file(GLOB TEST_SOURCES "main.cu")

add_executable(dl_test ${TEST_SOURCES})

target_include_directories(dl_test
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
)

target_link_libraries(dl_test
        NeuZephyr
        curand
)

if(WIN32)
    add_custom_command(
            TARGET dl_test
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/NeuZephyr.dll
            ${CMAKE_BINARY_DIR}/bin/NeuZephyr.dll
    )
elseif(UNIX)
    set_target_properties(dl_test PROPERTIES
            BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
    )
endif()

set_target_properties(dl_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
