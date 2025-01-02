cmake_minimum_required(VERSION 3.18)
project(Simple_DL_Framework LANGUAGES CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /wd1394")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 1394")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

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
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(NeuZephyr
        PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        curand
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

install(TARGETS NeuZephyr
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/NeuZephyr
        DESTINATION include
)