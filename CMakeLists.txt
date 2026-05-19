# 1. 指定最低 CMake 版本要求
cmake_minimum_required(VERSION 3.16)

# 【关键点 1】强行指定 nvcc 路径（请确保这真的是你 which nvcc 查出来的路径！）
set(CMAKE_CUDA_COMPILER "/workspace/whx/cuda-12.1/bin/nvcc")

# 【关键点 2】把架构声明提前到 project 之前，防止某些 CMake 版本发神经
set(CMAKE_CUDA_ARCHITECTURES "86")

# 2. 定义工程名称，正式开启异构双语支持
project(hxinfer LANGUAGES CXX CUDA)

# 确保 CUDA 也是 C++17 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =====================================================================
# 🚨 紧急避险：彻底关闭 Unity Build！
# Unity Build 会把多个代码文件强行拼成一个大文件编译。
# 刚引入 CUDA 时，极容易引发内部函数的“Multiple definition（重复定义）”等一大堆玄学报错。
# =====================================================================
set(CMAKE_UNITY_BUILD OFF)
# set(CMAKE_UNITY_BUILD_BATCH_SIZE 100) # 这行也注释掉作废

# 寻找本地安装的 CUDA 工具链
find_package(CUDAToolkit REQUIRED)

# 3. 强制使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. 开启暴力优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")
# 暂时先关掉跨文件优化，等 CUDA 跑通了再开，减少干扰
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# 5. 告诉编译器“菜单”在哪里
include_directories(${CMAKE_SOURCE_DIR}/include)

# =====================================================================
# 核心源码收集区
# =====================================================================
# 6. 自动收集“后厨”的源文件
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE CU_FILES "src/*.cu")

# =====================================================================
# 🛠️ 架构大升级：将核心源码打包成静态库 (STATIC Library)
# =====================================================================
# 7. 把所有的算子、分配器、张量，炼制成一个纯粹的底层引擎库：hxinfer_core
add_library(hxinfer_core STATIC
        ${SRC_FILES}
        ${CU_FILES}
)

# 8. 为引擎库装配底层依赖（CUDA 和 数学库）
target_link_libraries(hxinfer_core
        m
        CUDA::cudart
        CUDA::cublas
        sentencepiece
)

# =====================================================================
# 🚀 主力执行程序：点火入口
# =====================================================================
# 9. 主引擎现在变得极其轻量，它只包含入口的 main.cpp
add_executable(hxinfer_engine main.cpp)

# 10. 将核心引擎库“装载”到主引擎中
target_link_libraries(hxinfer_engine hxinfer_core)

# =====================================================================
# 🧪 自动化测试引擎 (Demos)
# =====================================================================
# 1. 自动收集 demos 目录下所有的 .cpp 和 .cu 文件
file(GLOB DEMO_SOURCES "demos/*.cpp" "demos/*.cu")

# 2. 开启 for 循环，遍历每一个 Demo
foreach(demo_src ${DEMO_SOURCES})

    # 3. 提取纯文件名作为 Target 名称
    get_filename_component(demo_name ${demo_src} NAME_WE)

    # 4. 注册可执行目标
    add_executable(${demo_name} ${demo_src})

    # 5. 【修复核心】所有的 Demo 现在链接的是 hxinfer_core 库！
    # 同时加上 CUDA 运行时，防止底层 API 找不到
    target_link_libraries(${demo_name}
            hxinfer_core
            CUDA::cudart
            CUDA::cublas
    )

endforeach()