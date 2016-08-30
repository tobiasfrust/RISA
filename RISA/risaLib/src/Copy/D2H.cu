/*
 *  Copyright 2016
 *
 *  D2H.cu
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#include <risa/Copy/D2H.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <ddrf/cuda/Check.h>
#include <ddrf/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <pthread.h>
#include <exception>

namespace risa {
namespace cuda {

D2H::D2H(const std::string& configFile) : reconstructionRate_(0), counter_(1.0){

   if (!readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::D2H: unable to read config file. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   memoryPoolIdx_ =
         ddrf::MemoryPool<hostManagerType>::instance()->registerStage(memPoolSize_,
               numberOfPixels_ * numberOfPixels_);

//   memoryPoolIdx_ =
//        ddrf::MemoryPool<hostManagerType>::instance()->registerStage(memPoolSize_,
//               432*500);

   //custom streams are necessary, because profiling with nvprof not possible with
   //-default-stream per-thread option
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      streams_[i] = stream;
   }

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &D2H::processor, this, i };
   }
   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::D2H: Running " << numberOfDevices_ << " Threads.";
   tmr_.start();
}

D2H::~D2H() {
   BOOST_LOG_TRIVIAL(info) << "Reconstructed " << reconstructionRate_ << " Images/s in average.";
   ddrf::MemoryPool<hostManagerType>::instance()->freeMemory(memoryPoolIdx_);
   for(auto i = 0; i < numberOfDevices_; i++){
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
   }
}

auto D2H::process(input_type&& img) -> void {
   if (img.valid()) {
      if(img.index() == 0)
         tmr_.start();
      if((count_ % 10000) == 9999){
         tmr_.stop();
         reconstructionRate_ = (reconstructionRate_*(counter_-1.0) + 10000.0/(tmr_.elapsed())) / counter_;
         counter_ += 1.0;
         BOOST_LOG_TRIVIAL(info) << "Reconstructing at " << 10000.0/(tmr_.elapsed()) << " Images/second.";
         tmr_.start();
      }
      count_++;
      BOOST_LOG_TRIVIAL(debug)<< "Image " << img.index() << "from device " << img.device() << "arrived";
      imgs_[img.device()].push(std::move(img));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "cuda::D2H: Received sentinel, finishing.";

      //send sentinal to processor threads and wait 'til they're finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         imgs_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }

      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "cuda::D2H: Finished.";
   }
}

auto D2H::wait() -> output_type {
   return results_.take();
}

auto D2H::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "D2H");
   CHECK(cudaSetDevice(deviceID));
   BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::D2H: Running Thread for Device " << deviceID;
   while (true) {
      auto img = imgs_[deviceID].take();
      if (!img.valid()) {
         BOOST_LOG_TRIVIAL(debug)<< "invalid image arrived.";
         break;
      }

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::D2H: Copy sinogram " << img.index() << " from device " << img.device();

      //copy image from device to host
      auto ret = ddrf::MemoryPool<hostManagerType>::instance()->requestMemory(
            memoryPoolIdx_);
      CHECK(
            cudaMemcpyAsync(ret.container().get(), img.container().get(),
                  img.size() * sizeof(float), cudaMemcpyDeviceToHost, streams_[deviceID]));
      ret.setIdx(img.index());
      ret.setPlane(img.plane());
      CHECK(cudaStreamSynchronize(streams_[deviceID]));

      //wait until work on device is finished
      results_.push(std::move(ret));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::D2H: Copy sinogram " << img.index() << " from device " << img.device() << " finished.";
   }
}

auto D2H::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());

   if (configReader.lookupValue("numberOfPixels", numberOfPixels_)
         && configReader.lookupValue("memPoolSize_D2H", memPoolSize_))
      return true;
   else
      return false;

}

}
}
