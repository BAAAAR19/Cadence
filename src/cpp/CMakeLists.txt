cmake_minimum_required(VERSION 3.26)
project(cadence_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_include_directories(_core PRIVATE include)
target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic -Werror)

# Lands next to the pure-Python package as cadence._core.
install(TARGETS _core DESTINATION cadence)
