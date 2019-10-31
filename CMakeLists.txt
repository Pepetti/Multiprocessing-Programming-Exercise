cmake_minimum_required(VERSION 3.5)
project (MPSP)

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/code_opencl)

add_compile_options(-g -O0 -Wall -Wextra -std=c++14)
#add_compile_options(-O3 -std=c++14)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/code_opencl/*.cpp)
add_executable(filter ${SOURCES})
target_link_libraries(filter ${OpenCL_LIBRARY})