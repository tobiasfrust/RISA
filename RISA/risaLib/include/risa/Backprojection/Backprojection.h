/*
 * Copyright 2016
 *
 * Backprojection.h
 *
 *  Created on: 26.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef BACKPROJECTION_H_
#define BACKPROJECTION_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>

#include <thrust/device_vector.h>

#include <map>
#include <thread>
#include <array>

namespace risa {
namespace cuda {

   namespace detail{
      enum InterpolationType: short {
         neareastNeighbor,
         linear
      };
   }

class Backprojection {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>;

public:
   Backprojection(const std::string& configFile);

   ~Backprojection();

   auto process(input_type&& inp) -> void;
   auto wait() -> output_type;

protected:

private:
   std::map<int, ddrf::Queue<input_type>> sinograms_;
   ddrf::Queue<output_type> results_;

   std::map<int, std::thread> processorThreads_;

   int numberOfProjections_;
   int numberOfDetectors_;
   int numberOfPixels_;
   float rotationOffset_;

   int numberOfDevices_;
   int numberOfStreams_;

   detail::InterpolationType interpolationType_;

   //kernel execution coniguration
   int blockSize2D_;

   int memPoolSize_;

   std::vector<int> lastStreams_;

   std::map<int, cudaStream_t> streams_;

   std::vector<unsigned int> memoryPoolIdxs_;

   auto processor(const int deviceID, const int streamID) -> void;
   auto initCuSparse(const int deviceID) -> void;
   auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* BACKPROJECTION_H_ */
