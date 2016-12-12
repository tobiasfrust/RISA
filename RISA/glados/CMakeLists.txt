cmake_minimum_required(VERSION 3.5)

include_directories(
   ${BOOST_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
   "${CMAKE_SOURCE_DIR}/glados/include"
)

set(SOURCES
   "${CMAKE_SOURCE_DIR}/glados/src/Filesystem.cpp"
   "${CMAKE_SOURCE_DIR}/glados/src/observer/Subject.cpp"
)

set(LINK_LIBRARIES ${LINK_LIBRARIES}
   ${Boost_LIBRARIES} 
)

add_library(glados SHARED ${SOURCES})

target_link_libraries(glados ${LINK_LIBRARIES})
