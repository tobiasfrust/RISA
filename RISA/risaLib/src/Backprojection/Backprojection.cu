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
 * Authors: Tobias Frust <t.frust@hzdr.de>
 *
 */

#include <risa/Backprojection/Backprojection.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <ddrf/MemoryPool.h>
#include <ddrf/cuda/Coordinates.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

__constant__ float sinLookup[2048];
__constant__ float cosLookup[2048];
__constant__ float normalizationFactor[1];
__constant__ float scale[1];
__constant__ float imageCenter[1];

Backprojection::Backprojection(const std::string& configFile) {

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::Backprojection: Configuration file could not be loaded successfully. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //allocate memory in memory pool for each device
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      memoryPoolIdxs_.push_back(
         ddrf::MemoryPool<deviceManagerType>::instance()->registerStage(
               memPoolSize_, numberOfPixels_ * numberOfPixels_));
   }

   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      //custom streams are necessary, because profiling with nvprof seems to be
      //not possible with -default-stream per-thread option
      cudaStream_t stream;
      CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 2));
      streams_[i] = stream;
   }

   //initialize worker thread
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] =
         std::thread { &Backprojection::processor, this, i };
   }
   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Backprojection: Running " << numberOfDevices_ << " Threads.";
}

Backprojection::~Backprojection() {
   for (auto idx : memoryPoolIdxs_) {
      ddrf::MemoryPool<deviceManagerType>::instance()->freeMemory(idx);
   }
   for (auto& ele : streams_) {
      //CHECK(cudaSetDevice(ele.first/numberOfStreams_));
      CHECK(cudaStreamDestroy(ele.second));
   }
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Backprojection: Destroyed.";
}

auto Backprojection::process(input_type&& sinogram) -> void {
   if (sinogram.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "BP: Image arrived with Index: " << sinogram.index() << "to device " << sinogram.device();
      sinograms_[sinogram.device()].push(std::move(sinogram));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Backprojection: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         sinograms_[i].push(input_type());
      }
      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }

      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Backprojection: Finished.";
   }
}

auto Backprojection::wait() -> output_type {
   return results_.take();
}

auto Backprojection::processor(const int deviceID) -> void {
   CHECK(cudaSetDevice(deviceID));

   //init lookup tables for sin and cos
   std::vector<float> sinLookup_h(numberOfProjections_), cosLookup_h(
         numberOfProjections_);
   for (auto i = 0; i < numberOfProjections_; i++) {
      float theta = i * M_PI
            / (float) numberOfProjections_+ rotationOffset_ / 180.0 * M_PI;
      while (theta < 0.0) {
         theta += 2.0 * M_PI;
      }
      sincosf(theta, &sinLookup_h[i], &cosLookup_h[i]);
   }
   CHECK(
         cudaMemcpyToSymbol(sinLookup, sinLookup_h.data(),
               sizeof(float) * numberOfProjections_));
   CHECK(
         cudaMemcpyToSymbol(cosLookup, cosLookup_h.data(),
               sizeof(float) * numberOfProjections_));
   //constants for kernel
   const float scale_h = numberOfDetectors_ / (float) numberOfPixels_;
   const float normalizationFactor_h = M_PI / numberOfProjections_ / scale_h;
   const float imageCenter_h = (numberOfPixels_ - 1.0) * 0.5;
   CHECK(cudaMemcpyToSymbol(normalizationFactor, &normalizationFactor_h, sizeof(float)));
   CHECK(cudaMemcpyToSymbol(scale, &scale_h, sizeof(float)));
   CHECK(cudaMemcpyToSymbol(imageCenter, &imageCenter_h, sizeof(float)));
   dim3 blocks(blockSize2D_, blockSize2D_);
   dim3 grids(std::ceil(numberOfPixels_ / (float) blockSize2D_),
         std::ceil(numberOfPixels_ / (float) blockSize2D_));
   if(interpolationType_ == detail::InterpolationType::linear)
      CHECK(cudaFuncSetCacheConfig(backProjectLinear, cudaFuncCachePreferL1));
   else if(interpolationType_ == detail::InterpolationType::neareastNeighbor)
      CHECK(cudaFuncSetCacheConfig(backProjectNearest, cudaFuncCachePreferL1));
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::BP: Running Thread for Device " << deviceID;
   while (true) {
      //execution is blocked until next element arrives in queue
      auto sinogram = sinograms_[deviceID].take();
      //if sentinel, finish thread execution
      if (!sinogram.valid())
         break;

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Backprojection: Backprojecting sinogram with Index " << sinogram.index();

      //allocate device memory for reconstructed picture
      auto recoImage =
            ddrf::MemoryPool<deviceManagerType>::instance()->requestMemory(
                  memoryPoolIdxs_[deviceID]);

      if(useTextureMemory_){
         cudaResourceDesc resDesc;
         memset(&resDesc, 0, sizeof(resDesc));
         resDesc.resType = cudaResourceTypeLinear;
         resDesc.res.linear.devPtr = sinogram.container().get();
         resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
         resDesc.res.linear.desc.x = 32; // bits per channel
         resDesc.res.linear.sizeInBytes = sinogram.size()*sizeof(float);

         cudaTextureDesc texDesc;
         memset(&texDesc, 0, sizeof(texDesc));
         texDesc.addressMode[0] = cudaAddressModeBorder;
         texDesc.addressMode[1] = cudaAddressModeBorder;
         texDesc.normalizedCoords = false;

         // create texture object: we only have to do this once!
         cudaTextureObject_t tex=0;
         CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
         backProjectTex<<<grids, blocks, 0, streams_[deviceID]>>>(tex, recoImage.container().get(),
                             numberOfPixels_, numberOfProjections_, numberOfDetectors_);
         CHECK(cudaDestroyTextureObject(tex));
      }else{
         if(interpolationType_ == detail::InterpolationType::linear)
            backProjectLinear<<<grids, blocks, 0, streams_[deviceID]>>>(
                  sinogram.container().get(), recoImage.container().get(),
                  numberOfPixels_, numberOfProjections_, numberOfDetectors_);
         else if(interpolationType_ == detail::InterpolationType::neareastNeighbor)
            backProjectNearest<<<grids, blocks, 0, streams_[deviceID]>>>(
                  sinogram.container().get(), recoImage.container().get(),
                  numberOfPixels_, numberOfProjections_, numberOfDetectors_);
      }
      CHECK(cudaPeekAtLastError());

      recoImage.setIdx(sinogram.index());
      recoImage.setDevice(deviceID);
      recoImage.setPlane(sinogram.plane());
      recoImage.setStart(sinogram.start());

      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(recoImage));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Backprojection: Reconstructing sinogram with Index " << sinogram.index() << " finished.";
   }
}

auto Backprojection::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(
         configFile.data());
   std::string interpolationStr;
   if (configReader.lookupValue("numberOfParallelProjections", numberOfProjections_)
         && configReader.lookupValue("numberOfParallelDetectors", numberOfDetectors_)
         && configReader.lookupValue("numberOfPixels", numberOfPixels_)
         && configReader.lookupValue("rotationOffset", rotationOffset_)
         && configReader.lookupValue("blockSize2D_backProjection", blockSize2D_)
         && configReader.lookupValue("memPoolSize_backProjection", memPoolSize_)
         && configReader.lookupValue("interpolationType", interpolationStr)
         && configReader.lookupValue("useTextureMemory", useTextureMemory_)
         && configReader.lookupValue("backProjectionAngleTotal", backProjectionAngleTotal_)){
      if(interpolationStr == "nearestNeighbour")
         interpolationType_ = detail::InterpolationType::neareastNeighbor;
      else if(interpolationStr == "linear")
         interpolationType_ = detail::InterpolationType::linear;
      else{
         BOOST_LOG_TRIVIAL(warning) << "recoLib::cuda::Backprojection: Requested interpolation mode not supported. Using linear-interpolation.";
         interpolationType_ = detail::InterpolationType::linear;
      }

      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;
}

__global__ void backProjectLinear(const float* const __restrict__ sinogram,
         float* __restrict__ image,
         const int numberOfPixels,
         const int numberOfProjections,
         const int numberOfDetectors){

   const auto x = ddrf::cuda::getX();
   const auto y = ddrf::cuda::getY();

   float sum = 0.0;

   if(x >= numberOfPixels || y >= numberOfPixels)
      return;

   const int centerIndex = numberOfDetectors * 0.5;

   const float xp = (x - imageCenter[0]) * scale[0];
   const float yp = (y - imageCenter[0]) * scale[0];

#pragma unroll 16
   for(auto projectionInd = 0; projectionInd < numberOfProjections; projectionInd++){
      const float t = xp * cosLookup[projectionInd] + yp * sinLookup[projectionInd];
      const int a = floor(t);
      const int aCenter = a + centerIndex;
      if(aCenter >= 0 && aCenter < numberOfDetectors){
         sum = sum + ((float)(a + 1) - t) * sinogram[projectionInd * numberOfDetectors + aCenter];
      }
      if((aCenter + 1) >= 0 && (aCenter + 1) < numberOfDetectors){
         sum = sum + (t - (float)a) * sinogram[projectionInd * numberOfDetectors + aCenter + 1];
      }

   }
   image[x + y * numberOfPixels] = sum * normalizationFactor[0];
}

__global__ void backProjectTex(cudaTextureObject_t tex,
         float* __restrict__ image,
         const int numberOfPixels,
         const int numberOfProjections,
         const int numberOfDetectors){

   const auto x = ddrf::cuda::getX();
   const auto y = ddrf::cuda::getY();

   float sum = 0.0;

   if(x >= numberOfPixels || y >= numberOfPixels)
      return;

   const float centerIndex = numberOfDetectors * 0.5;

   const float xp = (x - imageCenter[0]) * scale[0];
   const float yp = (y - imageCenter[0]) * scale[0];

#pragma unroll 8
   for(auto projectionInd = 0; projectionInd < numberOfProjections; projectionInd++){
      const int t = __fadd_rn(xp * cosLookup[projectionInd], yp * sinLookup[projectionInd] + centerIndex) ;
      const int tCenter = t + projectionInd*numberOfDetectors;
      if(t >= 0 && t < numberOfDetectors){
         float val = tex1Dfetch<float>(tex, tCenter);
         sum += val;
      }
   }
   image[x + y * numberOfPixels] = sum * normalizationFactor[0];
}


__global__ void backProjectNearest(const float* const __restrict__ sinogram,
      float* __restrict__ image, const int numberOfPixels,
      const int numberOfProjections, const int numberOfDetectors) {

   const auto x = ddrf::cuda::getX();
   const auto y = ddrf::cuda::getY();

   if (x >= numberOfPixels || y >= numberOfPixels)
      return;

   float sum = 0.0;

   float *p_cosLookup = cosLookup;
   float *p_sinLookup = sinLookup;

   const float scale = numberOfPixels / (float) numberOfDetectors;
   const int centerIndex = numberOfDetectors * 0.5;

   const float xp = (x - (numberOfPixels - 1.0) * 0.5) / scale;
   const float yp = (y - (numberOfPixels - 1.0) * 0.5) / scale;

#pragma unroll 16
   for (auto projectionInd = 0; projectionInd < numberOfProjections;
         projectionInd++) {
      //const int t = round(xp * cosLookup[projectionInd] + yp * sinLookup[projectionInd]) + centerIndex;
      const int t = round(xp * *p_cosLookup + yp * *p_sinLookup) + centerIndex;
      ++p_cosLookup; ++p_sinLookup;
      if (t >= 0 && t < numberOfDetectors)
         sum += sinogram[projectionInd * numberOfDetectors + t];
   }
   image[x + y * numberOfPixels] = sum * M_PI / numberOfProjections * scale;
}

}
}
