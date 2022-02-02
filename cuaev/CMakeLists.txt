cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cuaev LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Define our library target
add_library(cuaev SHARED cuaev.cpp aev.cu aev.h cuaev_cub.cuh)
target_include_directories(cuaev PRIVATE ./)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(cuaev ${TORCH_LIBRARIES})
target_link_libraries(cuaev ${PYTHON_LIBRARIES})
