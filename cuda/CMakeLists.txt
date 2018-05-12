#project(test)
find_package(CUDA)

file( GLOB  cu  *.cu)
file( GLOB  hdr *.hpp *.h )


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

CUDA_ADD_LIBRARY(mvs_cuda ${cu} ${hdr})
SET(CUDA_NVCC_FLAGS --compiler-options "-std=c++03";-arch=sm_20)
#ADD_SUBDIRECTORY(util)
#target_link_libraries(mvs_cuda util_cuda util)

if (CUDA_FOUND)
	message("========================================================")
	message("cuda is found.The dirs is:")
	MESSAGE(${CUDA_INCLUDE_DIRS})
	message("========================================================")
	include_directories("${CUDA_INCLUDE_DIRS}")
	#target_link_libraries(mvs_cuda ${CUDA_INCLUDE_DIRS})
else()
	message("-- Can't find CUDA. Continuing without it.")
endif()

