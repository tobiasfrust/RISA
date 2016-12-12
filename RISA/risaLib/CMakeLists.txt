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

include_directories(
   ${LIBCONFIGPP_INCLUDE_DIR} 
   ${BOOST_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
   "${CMAKE_SOURCE_DIR}/risaLib/include"
   "${CMAKE_SOURCE_DIR}/glados/include"
)

set(SOURCES
   "${CMAKE_SOURCE_DIR}/risaLib/src/ConfigReader/ConfigReader.cpp"
   "${CMAKE_SOURCE_DIR}/risaLib/src/DetectorInterpolation/DetectorInterpolation.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Reordering/Reordering.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Filter/Filter.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Copy/D2H.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Copy/H2D.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Fan2Para/Fan2Para.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Backprojection/Backprojection.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Masking/Masking.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Attenuation/Attenuation.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Loader/OfflineLoader.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Loader/OfflineLoader_perfTest.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/Saver/OfflineSaver.cu"
   "${CMAKE_SOURCE_DIR}/risaLib/src/UDPReceiver/Receiver/Receiver.cpp"
   "${CMAKE_SOURCE_DIR}/risaLib/src/UDPReceiver/ReceiverModule/ReceiverModule.cpp"
   "${CMAKE_SOURCE_DIR}/risaLib/src/UDPReceiver/UDPServer/UDPServer.cpp"
   "${CMAKE_SOURCE_DIR}/risaLib/src/template/Template.cu"
)

set(LINK_LIBRARIES ${LINK_LIBRARIES}
   ${LIBCONFIGPP_LIBRARY}
   ${CUDA_cusparse_LIBRARY}
   ${Boost_LIBRARIES} 
   ${TIFF_LIBRARIES}
)

CUDA_ADD_LIBRARY(RISA ${SOURCES} SHARED)

CUDA_ADD_CUFFT_TO_TARGET(RISA)
target_link_libraries(RISA ${LINK_LIBRARIES})
