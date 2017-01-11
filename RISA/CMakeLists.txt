# This file is part of the RISA library
#
# Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
#
# RISA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RISA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ddafa. If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)
project(RISA)

#define some speciel output paths for libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

#use c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_USE_MULTITHREADED ON)
SET(BOOST_MIN_VERSION "1.58.0")
add_definitions(-DBOOST_LOG_DYN_LINK=1)

SET(CUDA_MIN_VERSION "7.5")

#find required packages
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS system log filesystem program_options REQUIRED)
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)
find_package(TIFF REQUIRED)
find_package(OpenMP)
find_package(Doxygen)

#for cuda 7.5 and gcc5 add workaround (temporary)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
		if(CUDA_VERSION_MAJOR LESS 8.0)
			message("identified gcc-5 --> need to apply workaround")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -D_FORCE_INLINES")
		endif()
	endif()
endif()

#tell executable where to find the libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=../lib")

#add doc generation
if(DOXYGEN_FOUND)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
   add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

#add openmp
if(OPENMP_FOUND)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# nvcc flags
list(APPEND CUDA_NVCC_FLAGS "-O3 -use_fast_math")
#list(APPEND CUDA_NVCC_FLAGS "-lineinfo")
#list(APPEND CUDA_NVCC_FLAGS "--maxrregcount=16")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
if(NOT(CUDA_VERSION_MAJOR LESS 8.0))
   list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")
endif()
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall")

#descend into subdirectories
add_subdirectory(glados)
add_subdirectory(risaLib)
add_subdirectory(example)
