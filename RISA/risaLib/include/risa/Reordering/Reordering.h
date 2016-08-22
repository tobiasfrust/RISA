/*
 * Copyright 2016
 *
 * Reordering.h
 *
 *  Created on: 09.08.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef REORDERING_H_
#define REORDERING_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>

namespace risa {
namespace cuda {

class Reordering {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

public:
   Reordering(const std::string& configFile);
   ~Reordering();

   auto process(input_type&& img) -> void;
   auto wait() -> output_type;

private:

   std::map<int, ddrf::Queue<input_type>> sinos_;
   ddrf::Queue<output_type> results_;

   std::map<int, std::thread> processorThreads_;
   std::map<int, cudaStream_t> streams_;

   std::map<int, unsigned int> memoryPoolIdxs_;

   auto processor(int deviceID) -> void;
   auto createHashTable(std::vector<int>& hashTable) -> void;

   int numberOfDevices_;

   int numberOfDetectorsPerModule_;
   int numberOfFanDetectors_;
   int numberOfFanProjections_;
   int memPoolSize_;

   auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* REORDERING_H_ */
