# setze CMake Version 3.0 voraus
cmake_minimum_required(VERSION 3.16...3.27)

# C++11 Standard verwenden
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Projektname fuer CMake
project(Proku)

# die Liste aller ausfuehrbaren Programme
add_executable(main  main.cpp)