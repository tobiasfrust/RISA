cmake_minimum_required(VERSION 3.5)

include_directories(
   ${LIBCONFIGPP_INCLUDE_DIR}
   ${BOOST_INCLUDE_DIRS}
   ${TIFF_INCLUDE_DIR}
   "${CMAKE_SOURCE_DIR}/risa/include"
   "${CMAKE_SOURCE_DIR}/ddrf/include"
)

set(SOURCES
   "${CMAKE_SOURCE_DIR}/risa/src/ConfigReader/ConfigReader.cpp"
   "${CMAKE_SOURCE_DIR}/risa/src/Filter/Filter.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Copy/D2H.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Copy/H2D.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Fan2Para/Fan2Para.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Backprojection/Backprojection.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/CropImage/CropImage.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Attenuation/Attenuation.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Loader/OfflineLoader.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/Saver/OfflineSaver.cu"
   "${CMAKE_SOURCE_DIR}/risa/src/UDPReceiver/Receiver/Receiver.cpp"
   "${CMAKE_SOURCE_DIR}/risa/src/UDPReceiver/ReceiverModule/ReceiverModule.cpp"
   "${CMAKE_SOURCE_DIR}/risa/src/UDPReceiver/UDPServer/UDPServer.cpp"
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
