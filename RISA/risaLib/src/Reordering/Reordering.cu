/*
 * Copyright 2016
 *
 * CropImage.cu
 *
 *  Created on: 31.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include <risa/Reordering/Reordering.h>
#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Basics/performance.h>

#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

__global__ void reorder(const unsigned short* __restrict__ unorderedSino, unsigned short* __restrict__ orderedSino,
      const int* __restrict__ hashTable, const int numberOfProjections, const int numberOfDetectors);

Reordering::Reordering(const std::string& configFile) {

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::CropImage: Configuration file could not be loaded successfully. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //custom streams are necessary, because profiling with nvprof not possible with
   //-default-stream per-thread option
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      //register in memory pool
      memoryPoolIdxs_[i] = ddrf::MemoryPool<deviceManagerType>::instance()->registerStage(memPoolSize_, numberOfFanDetectors_*numberOfFanProjections_);
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      streams_[i] = stream;
   }

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &Reordering::processor, this, i };
   }
   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: Running " << numberOfDevices_ << " Threads.";
}

Reordering::~Reordering() {
   for (auto idx : memoryPoolIdxs_) {
      CHECK(cudaSetDevice(idx.first));
      ddrf::MemoryPool<deviceManagerType>::instance()->freeMemory(idx.second);
   }
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
   }
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::CropImage: Destroyed.";
}

auto Reordering::process(input_type&& img) -> void {
   if (img.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "CropImage: Image arrived with Index: " << img.index() << "to device " << img.device();
      sinos_[img.device()].push(std::move(img));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         sinos_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }
      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::CropImage: Finished.";
   }
}

auto Reordering::wait() -> output_type {
   return results_.take();
}

/**
 * The processor()-Method takes one sinogram from the queue. Via the cuFFT-Library
 * it is transformed into frequency space for applying the filter function.
 * After filtering the transformation is reverted via the inverse fourier transform.
 * Finally, the filtered sinogram is pushed back into the output queue for
 * further processing.
 *
 */
auto Reordering::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "CropImage");
   CHECK(cudaSetDevice(deviceID));
   dim3 blocks(16, 16);
   dim3 grids(std::ceil(numberOfFanDetectors_/16.0),
         std::ceil(numberOfFanProjections_/16.0));

   std::vector<int> hashTable(numberOfFanDetectors_*numberOfFanProjections_);
   createHashTable(hashTable);

   auto d_hashTable = ddrf::cuda::make_device_ptr<int>(numberOfFanDetectors_*numberOfFanProjections_);
   CHECK(cudaMemcpy(d_hashTable.get(), hashTable.data(), sizeof(int)*hashTable.size(), cudaMemcpyHostToDevice));

   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Reordering: Running Thread for Device " << deviceID;
   while (true) {
      auto img = sinos_[deviceID].take();
      if (!img.valid())
         break;
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Reordering: Reordering image with Index " << img.index();

      auto sino_ordered = ddrf::MemoryPool<deviceManagerType>::instance()->requestMemory(memoryPoolIdxs_[deviceID]);

      reorder<<<grids, blocks, 0, streams_[deviceID]>>>(img.container().get(), sino_ordered.container().get(), d_hashTable.get(), numberOfFanProjections_, numberOfFanDetectors_);
      CHECK(cudaPeekAtLastError());

      sino_ordered.setIdx(img.index());
      sino_ordered.setDevice(img.device());
      sino_ordered.setPlane(img.plane());

      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(sino_ordered));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Reordering: Reordering image with Index " << img.index() << " finished.";
   }
}

/**
 * All values needed for setting up the class are read from the config file
 * in this function.
 *
 * @param[in] configFile path to config file
 *
 * @return returns true, if configuration file could be read successfully, else false
 */
auto Reordering::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());
   int samplingRate, scanRate;
   if (configReader.lookupValue("numberOfFanDetectors", numberOfFanDetectors_)
         && configReader.lookupValue("memPoolSize_Reordering", memPoolSize_)
         && configReader.lookupValue("samplingRate", samplingRate)
         && configReader.lookupValue("scanRate", scanRate)){
      numberOfDetectorsPerModule_ = 16;
      numberOfFanProjections_ = samplingRate * 1000000 / scanRate;
      return EXIT_SUCCESS;
   }
   else
      return EXIT_FAILURE;

}

auto Reordering::createHashTable(std::vector<int>& hashTable) -> void {
   int numberOfModules = 27;
   int i = 0;
   hashTable.resize(numberOfFanProjections_*numberOfFanDetectors_);
   for(auto projInd = 0; projInd < numberOfFanProjections_; projInd++){
      for(auto modInd = 0; modInd < numberOfModules; modInd++){
         for(auto detInd = 0; detInd < numberOfDetectorsPerModule_; detInd++){
            int index = detInd + projInd * numberOfDetectorsPerModule_ + modInd * numberOfDetectorsPerModule_*numberOfFanProjections_;
            hashTable[i] = index;
            i++;
         }
      }
   }
}

__global__ void reorder(const unsigned short* __restrict__ unorderedSino, unsigned short* __restrict__ orderedSino,
      const int* __restrict__ hashTable, const int numberOfProjections, const int numberOfDetectors) {
   const auto x = ddrf::cuda::getX();
   const auto y = ddrf::cuda::getY();
   if (x >= numberOfDetectors || y >= numberOfProjections)
      return;

   const int index = x + y * numberOfDetectors;
   orderedSino[index] = unorderedSino[hashTable[index]];
}

}
}

