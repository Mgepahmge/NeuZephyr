cmake_minimum_required(VERSION 3.30)
project(nz_cpp_test)

set(CMAKE_CXX_STANDARD 17)

link_directories(lib)
include_directories(include/NeuZephyr)
add_executable(nz_cpp_test main.cpp)
target_link_libraries(nz_cpp_test NeuZephyr)
