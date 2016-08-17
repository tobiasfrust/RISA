/*
 *  Copyright 2016
 *
 *  H2D.cu
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#include <risa/Copy/H2D.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

H2D::H2D(const std::string& configFile) : lastDevice_{0}, worstCaseTime_{0.0}, bestCaseTime_{std::numeric_limits<double>::max()},
      lastIndex_{0u}, lostSinos_{0u}{

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "Configuration file could not be read. Please check!");
   }
   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //allocate memory on all available devices
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      memoryPoolIdxs_[i] =
            ddrf::MemoryPool<deviceManagerType>::instance()->registerStage(memPoolSize_,
                  numberOfDetectors_ * numberOfProjections_);
      //custom streams are necessary, because profiling with nvprof seems to be
      //not possible with -default-stream per-thread option
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      streams_[i] = stream;
   }

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &H2D::processor, this, i };
   }

   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::H2D: Running " << numberOfDevices_ << " Threads.";
}

H2D::~H2D() {
   for (auto idx : memoryPoolIdxs_) {
      CHECK(cudaSetDevice(idx.first));
      ddrf::MemoryPool<deviceManagerType>::instance()->freeMemory(idx.second);
   }
   for(auto i = 0; i < numberOfDevices_; i++){
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
   }
   BOOST_LOG_TRIVIAL(info) << "WorstCaseTime: " << worstCaseTime_ << "s; BestCaseTime: " << bestCaseTime_ << "s;";
   BOOST_LOG_TRIVIAL(info) << "Could not reconstruct " << lostSinos_ << " elements; " << lostSinos_/(double)lastIndex_*100.0 << "% loss";
}

/**
 *
 *
 *
 *
 */
auto H2D::process(input_type&& sinogram) -> void {
   if (sinogram.valid()) {
      if(sinogram.index() > 0)
         tmr_.stop();
      BOOST_LOG_TRIVIAL(debug) << "H2D: Image arrived with Index: " << sinogram.index() << "to device " << lastDevice_;
      sinograms_[lastDevice_].push(std::move(sinogram));
      lastDevice_ = (lastDevice_ + 1) % numberOfDevices_;
      double time = tmr_.elapsed();
      if(sinogram.index() > 0){
         if(time < bestCaseTime_)
            bestCaseTime_ = time;
         if(time > worstCaseTime_)
            worstCaseTime_ = time;
      }
      tmr_.start();
      int diff = sinogram.index() - lastIndex_ - 1;
      lostSinos_ += diff;
      if(diff > 0)
         BOOST_LOG_TRIVIAL(debug) << "Skipping " << diff << " elements.";
      if(count_%10000 == 0)
         BOOST_LOG_TRIVIAL(info) << "Lost " << lostSinos_ << " elements; " << lostSinos_/(double)lastIndex_*100.0 << "% loss";
      count_++;
      lastIndex_ = sinogram.index();
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::H2D: Received sentinel, finishing.";

      //send sentinel to all processor threads and wait 'til they're finished
      for(auto i = 0; i < numberOfDevices_; i++){
         sinograms_[i].push(input_type());
      }

      //wait until all threads are finished
      for(auto i = 0; i < numberOfDevices_; i++){
         processorThreads_[i].join();
      }

      //push sentinel to results for next stage
      results_.push(output_type());

      BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::H2D: Finished.";
   }
}

auto H2D::wait() -> output_type {
   return results_.take();
}

/**
 * Anytime, when there is a new input image in the input queue
 * this function takes it and transfers it asynchronously to the device.
 * No Memory Allocation is needed, due to use of MemoryPool.
 * Thus, no cudaFree or cudaMalloc is performed and device will not be loced.
 * Finally, the image is pushed into the output queue for further processing.
 */
auto H2D::processor(int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "H2D");
   CHECK(cudaSetDevice(deviceID));
   //for conversion from short to float
   std::vector<float> temp(numberOfProjections_*numberOfDetectors_);
   auto inputShort_d = ddrf::cuda::make_device_ptr<unsigned short>(numberOfProjections_*numberOfDetectors_);
   BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::H2D: Running Thread for Device " << deviceID;
   while (true) {
      auto sinogram = sinograms_[deviceID].take();
      if (!sinogram.valid())
         break;

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::H2D: Copy sinogram " << sinogram.index() << " to device " << deviceID;

      //copy image from device to host
      auto img = ddrf::MemoryPool<deviceManagerType>::instance()->requestMemory(
            memoryPoolIdxs_[deviceID]);

      CHECK(
            cudaMemcpyAsync(img.container().get(),sinogram.container().get(),
                   sinogram.size() * sizeof(unsigned short), cudaMemcpyHostToDevice, streams_[deviceID]));

      //needs to be set due to reuse of memory
      img.setIdx(sinogram.index());
      img.setDevice(deviceID);
      img.setPlane(sinogram.plane());

      CHECK(cudaStreamSynchronize(streams_[deviceID]));

      //wait until work on device is finished
      results_.push(std::move(img));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::H2D: Copy sinogram " << sinogram.index() << " to device finished.";
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
auto H2D::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());
   int samplingRate, scanRate;
   if (configReader.lookupValue("numberOfFanDetectors", numberOfDetectors_)
         && configReader.lookupValue("memPoolSize_H2D", memPoolSize_)
         && configReader.lookupValue("samplingRate", samplingRate)
         && configReader.lookupValue("scanRate", scanRate)){
      numberOfProjections_ = samplingRate * 1000000 / scanRate;
      return EXIT_SUCCESS;
   }
   else
      return EXIT_FAILURE;

}

}
}
