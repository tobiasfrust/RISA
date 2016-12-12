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
   "${CMAKE_SOURCE_DIR}/glados/include"
   "${CMAKE_SOURCE_DIR}/risaLib/include"
   ${Boost_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
)

link_libraries(
   ${Boost_LIBRARIES}
   ${TIFF_LIBRARIES}   
   glados
   RISA
)

set(SOURCES
   main.cpp
)

CUDA_ADD_EXECUTABLE(example ${SOURCES})

target_link_libraries(example ${LIBRARIES})
