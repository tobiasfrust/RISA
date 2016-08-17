cmake_minimum_required(VERSION 3.5)

include_directories(
   "${CMAKE_SOURCE_DIR}/ddrf/include"
   "${CMAKE_SOURCE_DIR}/risa/include"
   ${Boost_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
)

link_libraries(
   ddrf
   RISA
   ${Boost_LIBRARIES}
   ${TIFF_LIBRARIES}
)

set(SOURCES
   main.cu
)

CUDA_ADD_EXECUTABLE(risaExample ${SOURCES})

target_link_libraries(risaExample ${LIBRARIES})
