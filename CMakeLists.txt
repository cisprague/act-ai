cmake_minimum_required(VERSION 2.8)

find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/hpp)

add_subdirectory(${CMAKE_SOURCE_DIR}/cpp)
