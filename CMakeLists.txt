cmake_minimum_required(VERSION 3.10)

# set the project name
project(icp)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

########################################
# CUDA Setup
########################################
message(${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

set(CUDA_VERBOSE_BUILD ON)

find_package(PCL REQUIRED)
include_directories(
  ${PCL_INCLUDE_DIRS}
)

include_directories(
#icp_gpu
)

# add the executable
add_executable(icp icp_pcl.cpp)

target_link_libraries(icp
${PCL_LIBRARIES}
)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
# add the executable
cuda_add_executable(icp_gpu 
icp_gpu/main.cpp
icp_gpu/kernel.cu
)

target_link_libraries(icp_gpu
${PCL_LIBRARIES}
)

# add the library 
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
cuda_add_library(feezhu_icp
icp_gpu/kernel.cu
)

target_link_libraries(feezhu_icp
${PCL_LIBRARIES}
)

target_include_directories(feezhu_icp PUBLIC
icp_gpu
)
