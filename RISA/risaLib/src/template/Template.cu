/*
 *  Copyright 2016
 *
 *  Template.cu
 *
 *  Created on: 13.11.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#include <risa/template/Template.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <ddrf/cuda/Check.h>
#include <ddrf/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <pthread.h>
#include <exception>

namespace risa {
namespace cuda {

Template::Template(const std::string& configFile){

   if (!readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::Template: unable to read config file. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //when MemoryPool is required, register here:
   //memoryPoolIdx_ =
   //      ddrf::MemoryPool<hostManagerType>::instance()->registerStage(memPoolSize_,
   //            numberOfPixels_ * numberOfPixels_);

   //custom streams are necessary, because profiling with nvprof not possible with
   //-default-stream per-thread option
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      cudaStream_t stream;
      CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));
      streams_[i] = stream;
   }

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &Template::processor, this, i };
   }

   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Template: Running " << numberOfDevices_ << " Threads.";
}

Template::~Template() {
   //when Memorypool was used, free memory here
   //ddrf::MemoryPool<hostManagerType>::instance()->freeMemory(memoryPoolIdx_);
   //when use of cudaStreams, destroy them here
   //for(auto i = 0; i < numberOfDevices_; i++){
   //   CHECK(cudaSetDevice(i));
   //   CHECK(cudaStreamDestroy(streams_[i]));
   //}
}

auto Template::process(input_type&& img) -> void {
   if (img.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "risa::cuda::Template: Image " << img.index() << "from device " << img.device() << "arrived";
      imgs_[img.device()].push(std::move(img));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "risa::cuda::Template: Received sentinel, finishing.";

      //send sentinal to processor threads and wait 'til they're finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         imgs_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }

      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "risa::cuda::Template: Finished.";
   }
}

auto Template::wait() -> output_type {
   return results_.take();
}

auto Template::processor(const int deviceID) -> void {
   CHECK(cudaSetDevice(deviceID));
   BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Template: Running Thread for Device " << deviceID;
   while (true) {
      auto img = imgs_[deviceID].take();
      if (!img.valid()) {
         BOOST_LOG_TRIVIAL(debug)<< "invalid image arrived.";
         break;
      }

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Template: ";

      //if necessary, request memory from MemoryPool here
      auto ret = ddrf::MemoryPool<hostManagerType>::instance()->requestMemory(
            memoryPoolIdx_);

      //<-- do work here -->

      //in case of a CUDA stage, synchronization needs to be done here
      //CHECK(cudaStreamSynchronize(streams_[deviceID]));

      //wait until work on device is finished
      results_.push(std::move(ret));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Template: ";
   }
}

auto Template::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());
   //e.g. reading the number of pixels in the reconstructed image from the given configuration file
   if (configReader.lookupValue("numberOfPixels", numberOfPixels_))
      return true;
   else
      return false;

}

}
}
