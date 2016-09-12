cmake_minimum_required(VERSION 3.5)

include_directories(
   "${CMAKE_SOURCE_DIR}/ddrf/include"
   "${CMAKE_SOURCE_DIR}/risaLib/include"
   ${Boost_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
)

link_libraries(
   ${Boost_LIBRARIES}
   ${TIFF_LIBRARIES}   
   ddrf
   RISA
)

set(SOURCES
   main.cpp
)

CUDA_ADD_EXECUTABLE(example ${SOURCES})

target_link_libraries(example ${LIBRARIES})
