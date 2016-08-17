/*
 * Copyright 2016
 *
 * CropImage.h
 *
 *  Created on: 31.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef CROPIMAGE_H_
#define CROPIMAGE_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>

namespace risa {
namespace cuda {

class CropImage {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;

public:
   CropImage(const std::string& configFile);
   ~CropImage();

   auto process(input_type&& img) -> void;
   auto wait() -> output_type;

private:

   std::map<int, ddrf::Queue<input_type>> imgs_;
   ddrf::Queue<output_type> results_;

   std::map<int, std::thread> processorThreads_;
   std::map<int, cudaStream_t> streams_;

   auto processor(int deviceID) -> void;

   int numberOfDevices_;

   int numberOfPixels_;


   auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* CROPIMAGE_H_ */
