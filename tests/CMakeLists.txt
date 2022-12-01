cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

# External dependencies
## OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Local
set(SOURCE_FILES
    main.cpp
	BruteforceNSearch.h
	BruteforceNSearch.cpp
	tests.cpp
	tests.h
	)
   
add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests PUBLIC TreeNSearch)
