set(INTERNAL_FILES
	source/internals/vectors_internals.h
	source/internals/octree_internals.h
	source/internals/shuffle_lut.h
)

set(SOURCE_FILES
	source/NeighborList.h
	source/TreeNSearch.h
	source/TreeNSearch.cpp
	)
    
# Create a library version so it can be linked with other modules
add_library(TreeNSearch ${INTERNAL_FILES} ${SOURCE_FILES} )
target_include_directories(TreeNSearch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)
target_include_directories(TreeNSearch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(FILES ${SOURCE_FILES} DESTINATION include/)
install(FILES ${INTERNAL_FILES} DESTINATION include/internals)

source_group("internals" FILES ${INTERNAL_FILES})
source_group("source" FILES ${SOURCE_FILES})
