cmake_minimum_required(VERSION 3.18)
project(Simple_DL_Framework VERSION 1.0.0 LANGUAGES CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for MSVC and non-MSVC
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /wd1394")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 1394")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Gather source and header files
file(GLOB_RECURSE LIB_SOURCES "src/*.cu")
file(GLOB_RECURSE LIB_HEADERS "include/NeuZephyr/*.cuh")

# Add the shared library target
add_library(NeuZephyr SHARED
        ${LIB_SOURCES}
        ${LIB_HEADERS}
)

# Include directories
target_include_directories(NeuZephyr
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(NeuZephyr
        PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        curand
)

# Ensure position-independent code (PIC) and separable compilation
set_target_properties(NeuZephyr PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Define compile-time export macros
target_compile_definitions(NeuZephyr
        PRIVATE NEUZEPHYR_EXPORTS
        PUBLIC NEUZEPHYR_DLL
)

# Installation rules
include(GNUInstallDirs)
install(TARGETS NeuZephyr
        EXPORT NeuZephyrTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(
        DIRECTORY include/NeuZephyr
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.cuh"
)

# Export targets
install(EXPORT NeuZephyrTargets
        FILE NeuZephyrTargets.cmake
        NAMESPACE NeuZephyr::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuZephyr
)

# Generate and install package config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/NeuZephyrConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeuZephyrConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/NeuZephyrConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuZephyr
)

install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/NeuZephyrConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/NeuZephyrConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuZephyr
)