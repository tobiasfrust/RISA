/*
 * Copyright 2016
 *
 * CropImage.cu
 *
 *  Created on: 31.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include <risa/Masking/Masking.h>
#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Basics/performance.h>

#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

//!   This CUDA kernel multiplies the mask and the reconstructed image
/**
 * @param[in,out] img            the reconstructed image, that is multiplied with the mask in-place
 * @param[in]     value          the value, the pixels shall be replaced with
 * @param[in]     numberOfPixels the number of pixels in the reconstruction grid in one dimension
 */
__global__ void mask(float* __restrict__ img, const float value, const int numberOfPixels);

Masking::Masking(const std::string& configFile) {

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::CropImage: Configuration file could not be loaded successfully. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

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
      processorThreads_[i] = std::thread { &Masking::processor, this, i };
   }
   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: Running " << numberOfDevices_ << " Threads.";
}

Masking::~Masking() {
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
   }
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::CropImage: Destroyed.";
}

auto Masking::process(input_type&& img) -> void {
   if (img.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "CropImage: Image arrived with Index: " << img.index() << "to device " << img.device();
      imgs_[img.device()].push(std::move(img));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         imgs_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }
      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::CropImage: Finished.";
   }
}

auto Masking::wait() -> output_type {
   return results_.take();
}

auto Masking::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "CropImage");
   CHECK(cudaSetDevice(deviceID));
   dim3 blocks(16, 16);
   dim3 grids(std::ceil(numberOfPixels_ / 16.0),
         std::ceil(numberOfPixels_ / 16.0));
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::CropImage: Running Thread for Device " << deviceID;
   while (true) {
      auto img = imgs_[deviceID].take();
      if (!img.valid())
         break;
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: CropImageing image with Index " << img.index();

      mask<<<grids, blocks, 0, streams_[deviceID]>>>(img.container().get(),
            0.0 ,numberOfPixels_);
      CHECK(cudaPeekAtLastError());

      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(img));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::CropImage: CropImageing image with Index " << img.index() << " finished.";
   }
}

auto Masking::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(
         configFile.data());
   if (configReader.lookupValue("numberOfPixels", numberOfPixels_))
      return EXIT_SUCCESS;

   return EXIT_FAILURE;
}

__global__ void mask(float* __restrict__ img, const float value, const int numberOfPixels) {
   const auto x = ddrf::cuda::getX();
   const auto y = ddrf::cuda::getY();
   if (x >= numberOfPixels || y >= numberOfPixels)
      return;
   const float center = (numberOfPixels - 1.0) * 0.5;
   const float dX = x - center;
   const float dY = y - center;
   const float distance = dX * dX + dY * dY;
   if (distance > numberOfPixels * numberOfPixels * 0.25) {
      img[x + numberOfPixels * y] = value;
   }
}

}
}

