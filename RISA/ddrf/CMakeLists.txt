cmake_minimum_required(VERSION 3.5)

include_directories(
   ${BOOST_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
   "${CMAKE_SOURCE_DIR}/ddrf/include"
)

set(SOURCES
   "${CMAKE_SOURCE_DIR}/ddrf/src/Filesystem.cpp"
   "${CMAKE_SOURCE_DIR}/ddrf/src/observer/Subject.cpp"
)

set(LINK_LIBRARIES ${LINK_LIBRARIES}
   ${Boost_LIBRARIES} 
)

add_library(ddrf SHARED ${SOURCES})

target_link_libraries(ddrf ${LINK_LIBRARIES})
