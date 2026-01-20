cmake_minimum_required(VERSION 3.10)

#Project Name
project(SpMV_Project)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI Package
find_package(MPI REQUIRED)

# Indclude source and header files
include_directories(src)

# Define the final executable name
# and list all the source/header files it needs.
add_executable(spmv_strong_exec
    src/main_strong.cpp
    src/Utils.cpp
    src/SparseMatrix.cpp
)

# Link MPI libraries top the executable.
target_link_libraries(spmv_strong_exec PRIVATE MPI::MPI_CXX)

# Optimization Flags for Release Builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -O3 for general optimization
    # -march=native to optimize for the host machine architecture
    target_compile_options(spmv_strong_exec PRIVATE -O3 -march=native)
endif()

add_executable(spmv_weak_exec
    src/main_weak.cpp      
    src/Utils.cpp          
    src/SparseMatrix.cpp   
)

# Link MPI libraries to the weak scaling executable.
target_link_libraries(spmv_weak_exec PRIVATE MPI::MPI_CXX)

# Optimization Flags for Release Builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(spmv_weak_exec PRIVATE -O3 -march=native)
endif()