cmake_minimum_required(VERSION 3.29)
project(Simple_DL_Framework CUDA)

# CUDA 设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}")
set(CMAKE_CXX_FLAGS "-fopenmp")

# 添加CUDA库目录
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# 设置库的源文件
file(GLOB_RECURSE LIB_SOURCES "src/*.cu")
file(GLOB_RECURSE LIB_HEADERS "include/DL-Framework/*.cuh")

# 创建库目标
add_library(dl_framework SHARED
        ${LIB_SOURCES}
        ${LIB_HEADERS}
)

# 设置库的包含目录
target_include_directories(dl_framework
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# 设置库的属性
set_target_properties(dl_framework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

target_compile_definitions(dl_framework
        PRIVATE DL_FRAMEWORK_EXPORTS
        PUBLIC DL_FRAMEWORK_DLL
)

# 链接CUDA库到dl_framework
target_link_libraries(dl_framework
        curand
)

# 创建测试可执行文件
add_executable(dl_test test/main.cu)

# 设置测试程序的包含目录
target_include_directories(dl_test
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# 链接测试程序与库
target_link_libraries(dl_test
        dl_framework
        curand
)

# 设置测试程序的输出目录
set_target_properties(dl_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则
install(TARGETS dl_framework
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/DL-Framework
        DESTINATION include
)