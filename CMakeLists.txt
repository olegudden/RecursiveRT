cmake_minimum_required(VERSION 3.16...3.27)

# 1. Project name MUST come first
project(Proku CXX)

# 2. Set C++17 Standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(external src)
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

# 3. Explicitly enforce C++17 directly on the executable target (Guarantees IntelliSense reads it)
target_compile_features(main PUBLIC cxx_std_17)