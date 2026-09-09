cmake_minimum_required(VERSION 3.26)
project(cadence LANGUAGES CXX)

# Week 3 adds src/cpp. Guarded so that `pip install -e .` works today, before
# the extension exists.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/CMakeLists.txt")
  add_subdirectory(src/cpp)
endif()
