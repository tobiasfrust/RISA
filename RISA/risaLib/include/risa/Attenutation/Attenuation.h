/*
 * Copyright 2016
 *
 * Attenuation.h
 *
 *  Created on: 02.06.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef ATTENUATION_H_
#define ATTENUATION_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>
#include <string>

namespace risa {
namespace cuda {

class Attenuation {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>;
public:
   Attenuation(const std::string& configFile);
   ~Attenuation();

   auto process(input_type&& img) -> void;
   auto wait() -> output_type;

private:

   std::map<int, ddrf::Queue<input_type>> sinograms_;
   ddrf::Queue<output_type> results_;

   std::map<int, std::thread> processorThreads_;
   std::map<int, cudaStream_t> streams_;
   std::map<int, unsigned int> memoryPoolIdxs_;

   auto processor(int deviceID) -> void;

   auto init() -> void;

   template <typename T>
   auto computeAverage(const std::vector<T>& values, std::vector<float>&average) -> void;

   template <typename T>
   auto readDarkInputFiles(std::string& file, std::vector<T>& values) -> void;

   template <typename T>
   auto readInput(std::string& path, std::vector<T>& values) -> void;

   template <typename T>
   auto relevantAreaMask(std::vector<T>& mask) -> void;

   int numberOfDevices_;

   //configuration values
   int numberOfDetectorModules_;
   int numberOfDetectors_;
   int numberOfProjections_;
   int numberOfPlanes_;
   int numberOfDarkFrames_;
   int numberOfRefFrames_;
   std::string pathDark_, pathReference_;

   //parameters for mask generation
   double sourceOffset_, lowerLimOffset_, upperLimOffset_;
   unsigned int xa_, xb_, xc_, xd_, xe_, xf_;

   //kernel execution coniguration
   int blockSize2D_;

   int memPoolSize_;

   //average values on host
   std::vector<float> avgDark_, avgReference_;

   auto readConfig(const std::string& configFile) -> bool;

};
}
}

#endif /* ATTENUATION_H_ */
