cmake_minimum_required(VERSION 3.10)
project(SpmvBenchmark CXX)

# Set the C++ standard to C++11 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenMP package (required for parallelization).
# This finds the compiler flags (like -fopenmp).
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Set the default build type to "Release".
# This enables optimizations (e.g., -O3) for performance.
set(CMAKE_BUILD_TYPE Release)

# Automatically find all .cpp files in the src/ directory.
file(GLOB SOURCES "src/*.cpp")
# Automatically find all .h files in the src/ directory.
file(GLOB HEADERS "src/*.h")

# Define the final executable name ('spmv_benchmark')
# and list all the source/header files it needs.
add_executable(spmv_benchmark ${SOURCES} ${HEADERS})

# Tell the compiler to look for headers in the 'src/' directory
# (e.g., for #include "Config.h").
target_include_directories(spmv_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
