/*
 *  Copyright 2016
 *
 *  Filter.cu
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#include <risa/Filter/Filter.h>
#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Basics/performance.h>
#include "cuda_kernels_filter.h"

#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Check.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

__constant__ float filter_d[2049];

__global__ void applyFilter(const int x, const int y, const float normalization, cufftComplex *data);

Filter::Filter(const std::string& configFile) {

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::Filter: Configuration file could not be loaded successfully. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //cuFFT library is initialized for each device
   for (auto i = 0; i < numberOfDevices_; i++) {
      initCuFFT(i);
   }

   designFilter();

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &Filter::processor, this, i };
   }
   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Filter: Running " << numberOfDevices_ << " Threads.";
}

Filter::~Filter() {
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
      CHECK_CUFFT(cufftDestroy(plansFwd_[i]));
      CHECK_CUFFT(cufftDestroy(plansInv_[i]));
   }
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Filter: Destroyed.";
}

auto Filter::process(input_type&& sinogram) -> void {
   if (sinogram.valid()) {
      BOOST_LOG_TRIVIAL(debug) << "Filter: Image arrived with Index: " << sinogram.index() << "to device " << sinogram.device();
      sinograms_[sinogram.device()].push(std::move(sinogram));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Filter: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         sinograms_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }
      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Filter: Finished.";
   }
}

auto Filter::wait() -> output_type {
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
auto Filter::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "Filter");
   CHECK(cudaSetDevice(deviceID));
   auto sinoFreq = ddrf::cuda::make_device_ptr<cufftComplex,
         ddrf::cuda::async_copy_policy>(
         numberOfProjections_ * ((numberOfDetectors_ / 2.0) + 1));
   dim3 dimBlock(blockSize2D_, blockSize2D_);
   dim3 dimGrid((int) ceil((numberOfDetectors_ / 2.0 + 1) / (float) blockSize2D_),
         (int) ceil(numberOfProjections_ / (float) blockSize2D_));
   CHECK(
         cudaMemcpyToSymbol(filter_d, filter_.data(), sizeof(float) * filter_.size()));
   const float normalizationFactor = 1.0/(float)numberOfDetectors_;
   BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Filter: Running Thread for Device " << deviceID;
   while (true) {
      auto sinogram = sinograms_[deviceID].take();
      if (!sinogram.valid())
         break;
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Filter: Filtering sinogram with Index " << sinogram.index();

      //forward transformation
      CHECK_CUFFT(
            cufftExecR2C(plansFwd_[deviceID],
                  (cufftReal* ) sinogram.container().get(),
                  thrust::raw_pointer_cast(&(sinoFreq[0]))));

      //Filtering
      applyFilter<<<dimGrid, dimBlock, 0, streams_[deviceID]>>>(
            (numberOfDetectors_ / 2) + 1, numberOfProjections_, normalizationFactor,
            thrust::raw_pointer_cast(&(sinoFreq[0])));

//      filterRamp<<<dimGrid, dimBlock, 0, streams_[deviceID]>>>(
//            (numberOfDetectors_ / 2) + 1, numberOfProjections_,
//            thrust::raw_pointer_cast(&(sinoFreq[0])));
      CHECK(cudaPeekAtLastError());

      //reverse transformation
      CHECK_CUFFT(
            cufftExecC2R(plansInv_[deviceID],
                  thrust::raw_pointer_cast(sinoFreq.get()),
                  (cufftReal* ) sinogram.container().get()));
      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(sinogram));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Filter: Filtering sinogram with Index " << sinogram.index() << " finished.";
   }
}

/**
 * Initializes the NVIDIA cuFFT Library for use in processor-Method
 * Creates a stream in which all operations in this class are executed.
 * This is necessary, because the cuFFT library ignores the compiler
 * option 'default-stream per-thread'.
 * Finally, memory for the transformed sinogram is allocated here.
 *
 */
auto Filter::initCuFFT(const int deviceID) -> void {

   CHECK(cudaSetDevice(deviceID));

   cudaStream_t stream;
   cufftHandle planFwd, planInv;

   CHECK(cudaStreamCreate(&stream));
   streams_[deviceID] = stream;

   CHECK_CUFFT(
         cufftPlanMany(&planFwd, 1, &numberOfDetectors_, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, numberOfProjections_));

   CHECK_CUFFT(cufftSetStream(planFwd, stream));

   CHECK_CUFFT(
         cufftPlanMany(&planInv, 1, &numberOfDetectors_, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, numberOfProjections_));

   CHECK_CUFFT(cufftSetStream(planInv, stream));

   plansFwd_[deviceID] = planFwd;
   plansInv_[deviceID] = planInv;
}

auto Filter::designFilter() -> void {
   int filterSize = numberOfDetectors_/2 + 1;
   filter_.reserve(filterSize);
   filter_.push_back(0.0);
   for(auto i = 1; i < filterSize; i++){
      //actual w at frequency axis
      const float w = 2 * M_PI * i / (float)numberOfDetectors_;
      if(w > M_PI*cutoffFraction_){
         filter_.push_back(0.0);
         continue;
      }
      float filterValue = 2 * i / (float)numberOfDetectors_; //* hanning(w, (float)1.0);
      if(filterType_ == detail::FilterType::hamming)
         filterValue *= hamming(w, cutoffFraction_);
      else if(filterType_ == detail::FilterType::hanning)
         filterValue *= hanning(w, cutoffFraction_);
      else if(filterType_ == detail::FilterType::sheppLogan)
         filterValue *= sheppLogan(w, cutoffFraction_);
      else if(filterType_ == detail::FilterType::cosine)
         filterValue *= cosine(w, cutoffFraction_);
      filter_.push_back(filterValue);
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
auto Filter::readConfig(const std::string& configFile) -> bool {
   recoLib::ConfigReader configReader = recoLib::ConfigReader(
         configFile.data());
   std::string filterType;
   if (configReader.lookupValue("numberOfParallelProjections", numberOfProjections_)
         && configReader.lookupValue("numberOfParallelDetectors", numberOfDetectors_)
         && configReader.lookupValue("numberOfPixels", numberOfPixels_)
         && configReader.lookupValue("blockSize2D_filter", blockSize2D_)
         && configReader.lookupValue("filterType", filterType)
         && configReader.lookupValue("cutoffFraction", cutoffFraction_)){
      if(filterType == "ramp")
         filterType_ = detail::FilterType::ramp;
      else if(filterType == "sheppLogan")
         filterType_ = detail::FilterType::sheppLogan;
      else if(filterType == "hamming")
         filterType_ = detail::FilterType::hamming;
      else if(filterType == "hanning")
         filterType_ = detail::FilterType::hanning;
      else if(filterType == "cosine")
         filterType_ = detail::FilterType::cosine;
      else{
         BOOST_LOG_TRIVIAL(error) << "recoLib::cuda::Filter: Requested filter mode not supported. Using Ramp-Filter.";
         filterType_ = detail::FilterType::ramp;
      }
      return EXIT_SUCCESS;
   }
   return EXIT_FAILURE;
}

__global__ void applyFilter(const int x, const int y, const float normalization, cufftComplex *data) {
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < x && j < y) {
      //cufft performs an unnormalized transformation ifft(fft(A))=length(A)*A
      //->normalization needs to be performed
      const float filterVal = filter_d[i] * normalization;
      data[i + j * x].x *= filterVal;
      data[i + j * x].y *= filterVal;
   }
}

}
}
