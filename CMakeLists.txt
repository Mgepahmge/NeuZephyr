cmake_minimum_required(VERSION 3.29)
project(Simple_DL_Framework CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2")
set(CMAKE_CXX_FLAGS "-fopenmp")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_executable(Simple_DL_Framework main.cu
        OperationKernels.cu
        OperationKernels.cuh
        Nodes.cu
        Nodes.cuh
        Tensor.cu
        Tensor.cuh)
target_link_libraries(Simple_DL_Framework curand)

set_target_properties(Simple_DL_Framework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
