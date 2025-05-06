cmake_minimum_required(VERSION 3.30)
project(NeuZephyrTest)

set(CMAKE_CXX_STANDARD 17)

link_directories(../cmake-build-release)
include_directories(../include/NeuZephyr)

enable_testing()
find_package(GTest REQUIRED)
add_executable(NeuZephyrTest Test.cpp)
add_custom_command(TARGET NeuZephyrTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/../cmake-build-release/NeuZephyr.dll"
        $<TARGET_FILE_DIR:NeuZephyrTest>
        COMMENT "Copying NeuZephyr.dll to output directory"
)
target_link_libraries(
        NeuZephyrTest
        GTest::GTest
        GTest::Main
        NeuZephyr
)
include(GoogleTest)
gtest_discover_tests(NeuZephyrTest)