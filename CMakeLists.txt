cmake_minimum_required(VERSION 3.16)

if(DEFINED ENV{CUDACXX} AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

project(hxinfer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_UNITY_BUILD OFF)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

include_directories(
    ${CMAKE_SOURCE_DIR}/hxinfer-engine/include
    ${CMAKE_SOURCE_DIR}/model_loaders/include
    ${CMAKE_SOURCE_DIR}/tokenizer/include
)

# ==================== Engine Core Library ====================
file(GLOB_RECURSE ENGINE_SRC "hxinfer-engine/src/*.cpp")
file(GLOB_RECURSE ENGINE_CU  "hxinfer-engine/src/*.cu")

add_library(hxinfer_core STATIC
    ${ENGINE_SRC}
    ${ENGINE_CU}
)

target_link_libraries(hxinfer_core
    m
    CUDA::cudart
    CUDA::cublas
    sentencepiece
)

# ==================== Main Engine ====================
add_executable(hxinfer_engine main.cpp)
target_link_libraries(hxinfer_engine hxinfer_core hxinfer_loaders hxinfer_tokenizer)

# ==================== Model Loaders ====================
file(GLOB_RECURSE LOADER_SRC "model_loaders/src/*.cpp")
add_library(hxinfer_loaders STATIC ${LOADER_SRC})
target_link_libraries(hxinfer_loaders hxinfer_core)

# ==================== Tokenizer ====================
file(GLOB_RECURSE TOKENIZER_SRC "tokenizer/src/*.cpp")
add_library(hxinfer_tokenizer STATIC ${TOKENIZER_SRC})
target_link_libraries(hxinfer_tokenizer sentencepiece)

# ==================== Demos ====================
file(GLOB DEMO_SOURCES "demos/*.cpp" "demos/*.cu")
foreach(demo_src ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_src} NAME_WE)
    add_executable(${demo_name} ${demo_src})
    target_link_libraries(${demo_name}
        hxinfer_core
        hxinfer_loaders
        hxinfer_tokenizer
        CUDA::cudart
        CUDA::cublas
    )
endforeach()

# ==================== Tests ====================
file(GLOB TEST_SOURCES "tests/unit/*.cpp" "tests/integration/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
        hxinfer_core
        hxinfer_loaders
        hxinfer_tokenizer
        CUDA::cudart
        CUDA::cublas
    )
endforeach()
