cmake_minimum_required(VERSION 3.24)
project(emaxim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(emaxim src/sycamore.cpp)

# set_target_properties(flower_core PROPERTIES OUTPUT_NAME "_core")

# Install into the Python package directory
install(TARGETS emaxim
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .)
