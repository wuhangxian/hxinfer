cmake_minimum_required(VERSION 3.22)
project(hxinfer)

set(CMAKE_CXX_STANDARD 17)

# 1. 寻找 CUDA
find_package(CUDA REQUIRED)
# ### 新增：寻找 glog 库 ###
find_package(glog REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
# -----------------------------------------------------------
# 🔥 关键修改：告诉编译器去项目的 include 文件夹下找头文件
# -----------------------------------------------------------
include_directories(include)

# 自动搜集源文件（可选，但推荐，方便以后加新文件）
# file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# 生成可执行文件
add_executable(hxinfer
        main.cpp
        # include/base/allocator.h  <-- 头文件其实不需要写在这里，只要上面有了 include_directories 即可
)

target_link_libraries(hxinfer
        ${CUDA_LIBRARIES}
        glog::glog
)

# === Playground ===
# 这一行是告诉编译器：我要把 playground/01_raw_memory.cpp 编译成一个叫 "step1" 的可执行程序
add_executable(step1 playground/01_raw_memory.cpp)