project(cuda_matcher LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(CUDA REQUIRED)



set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Ofast -pthread -march=native")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52 -gencode=arch=compute_86,code=sm_86 -O3 -march=native -pthread --use_fast_math")

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)


set(SOURCES src/main.cpp src/matcher.cu src/input.cpp)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(matcher ${SOURCES})

target_link_libraries(matcher ${CUDA_LIBRARIES} -lcublas -lcurand)
target_link_libraries(matcher argparse)

add_executable(test src/test.cpp src/matcher.cu src/input.cpp)
target_link_libraries(test ${CUDA_LIBRARIES} -lcublas -lcurand)
target_link_libraries(test argparse)



