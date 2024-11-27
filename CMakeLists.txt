cmake_minimum_required(VERSION 3.29)
project(Simple_DL_Framework CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}")
set(CMAKE_CXX_FLAGS "-fopenmp")

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

file(GLOB_RECURSE LIB_SOURCES "src/*.cu")
file(GLOB_RECURSE LIB_HEADERS "include/NeuZephyr/*.cuh")

add_library(NeuZephyr SHARED
        ${LIB_SOURCES}
        ${LIB_HEADERS}
)

target_include_directories(NeuZephyr
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_TOOLKIT_ROOT_DIR}/include
)

set_target_properties(NeuZephyr PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

target_compile_definitions(NeuZephyr
        PRIVATE NEUZEPHYR_EXPORTS
        PUBLIC NEUZEPHYR_DLL
)

target_link_libraries(NeuZephyr
        curand
)

install(TARGETS NeuZephyr
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/NeuZephyr
        DESTINATION include
)

add_subdirectory(test)
