set(INTERNAL_FILES
	source/internals/vectors_internals.h
	source/internals/octree_internals.h
	source/internals/shuffle_lut.h
)

set(INCLUDE_FILES
	source/NeighborList.h
	source/TreeNSearch.h
	)

set(SOURCE_FILES
	source/TreeNSearch.cpp
	)
    
# Create a library version so it can be linked with other modules
add_library(TreeNSearch ${INTERNAL_FILES} ${SOURCE_FILES} ${INCLUDE_FILES})
target_include_directories(TreeNSearch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)
target_include_directories(TreeNSearch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(FILES ${HEADER_FILES} DESTINATION include/)
install(FILES ${INTERNAL_FILES} DESTINATION include/internals)

install(TARGETS TreeNSearch
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

source_group("internals" FILES ${INTERNAL_FILES})
source_group("include" FILES ${INCLUDE_FILES})
source_group("source" FILES ${SOURCE_FILES})