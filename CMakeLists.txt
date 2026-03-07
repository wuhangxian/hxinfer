# 1. 指定最低 CMake 版本要求
cmake_minimum_required(VERSION 3.15)

# 2. 定义工程名称和支持的语言
# 【预警】你现在只切分 C++ 代码，所以只写 CXX。
# 【下午开工 CUDA 时】：把它改成 project(hxinfer LANGUAGES CXX CUDA)
project(hxinfer LANGUAGES CXX)

# 3. 强制使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. 开启暴力优化！-O3 是必须的，否则大模型推理会慢得像蜗牛
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")

# =====================================================================
# 核心目录配置区
# =====================================================================

# 5. 告诉编译器“菜单”在哪里：默认去 include/ 文件夹下找头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 6. 自动收集“后厨”的源文件 (不用你新增一个 cpp 就来这里改一行)
# 自动扫描 src/ 目录及其所有子目录下的所有 .cpp 文件
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# 【下午开工 CUDA 时】：把下面这行注释打开，自动扫描所有的 .cu 核函数文件
# file(GLOB_RECURSE CU_FILES "src/*.cu")

# =====================================================================
# 编译产物输出区
# =====================================================================

# 7. 生成最终的引擎执行程序
# 假设你的点火入口程序放到了 demo/main.cpp 里
add_executable(hxinfer_engine
        main.cpp
        ${SRC_FILES}
        # ${CU_FILES}  <-- 下午写 CUDA 时，把这行的注释也打开

)

# 8. 链接标准数学库 (Linux 环境下某些底层数学函数需要手动链接 -lm)
target_link_libraries(hxinfer_engine m)