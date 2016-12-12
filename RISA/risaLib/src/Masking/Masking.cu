/*
 * This file is part of the RISA-library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * RISA is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RISA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with RISA. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 30 November 2016
 * Authors: Tobias Frust (FWCC) <t.frust@hzdr.de>
 *
 */

#include <risa/Masking/Masking.h>
#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Basics/performance.h>

#include <glados/cuda/Launch.h>
#include <glados/cuda/Check.h>
#include <glados/cuda/Coordinates.h>

#include <boost/log/trivial.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

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
      CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 1));
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

      //normalization
      if(performNormalization_){
         auto pair = (thrust::minmax_element(thrust::device_pointer_cast(img.container().get()), thrust::device_pointer_cast(img.container().get()+img.size())));
         float min = *pair.first;
         float max = *pair.second;
         float diff = max- min;
         thrust::transform(thrust::device_pointer_cast(img.container().get()), thrust::device_pointer_cast(img.container().get()+img.size()), thrust::device_pointer_cast(img.container().get()), (thrust::placeholders::_1 - min)/diff);
      }
      mask<<<grids, blocks, 0, streams_[deviceID]>>>(img.container().get(),
            maskingValue_ ,numberOfPixels_);
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
   if (configReader.lookupValue("numberOfPixels", numberOfPixels_)
         && configReader.lookupValue("normalization", performNormalization_)
         && configReader.lookupValue("maskingValue", maskingValue_))
      return EXIT_SUCCESS;

   return EXIT_FAILURE;
}

__global__ void mask(float* __restrict__ img, const float value, const int numberOfPixels) {
   const auto x = glados::cuda::getX();
   const auto y = glados::cuda::getY();
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

