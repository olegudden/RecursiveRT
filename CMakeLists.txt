# setze CMake Version 3.0 voraus
cmake_minimum_required(VERSION 3.16...3.27)

# Projektname fuer CMake (MOVED UP)
project(Proku)

# C++11 Standard verwenden
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Externe Headerdateien
include_directories(external src)

# Sucht automatisch alle .cpp-Dateien im src-Ordner
file(GLOB_RECURSE SOURCES "src/*.cpp")

# die Liste aller ausfuehrbaren Programme (UPDATED TO INCLUDE ALL CPP FILES)
add_executable(main ${SOURCES})