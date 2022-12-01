cmake_minimum_required(VERSION 3.1)
project(TreeNSearch)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # Parallel compilation in Visual Studio
endif()

# External dependencies
## Enable SIMD instructions
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	add_compile_options("-march=native")
	add_compile_options("-mavx")
    add_compile_options("-Wno-ignored-attributes")  # AVX warnings
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	add_compile_options("/arch:AVX")
	add_compile_options("/wd4996 ")  # supress fopen warning
endif()

## OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Modules
add_subdirectory(TreeNSearch)
add_subdirectory(tests)
