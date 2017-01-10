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

#include "../../include/risa/Filter/Filter.h"
#include "../../include/risa/Basics/performance.h"
#include "cuda_kernels_filter.h"

#include <glados/cuda/Launch.h>
#include <glados/cuda/Check.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

__global__ void applyFilter(const int x, const int y, cufftComplex *data, const float* const __restrict__ filter);

Filter::Filter(const std::string& config_file) {

   risa::read_json config_reader{};
   config_reader.read(config_file);
   if (readConfig(config_reader)) {
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

auto Filter::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "Filter");
   CHECK(cudaSetDevice(deviceID));
   auto sinoFreq = glados::cuda::make_device_ptr<cufftComplex,
         glados::cuda::async_copy_policy>(
         numberOfProjections_ * ((numberOfDetectors_ / 2.0) + 1));
   dim3 dimBlock(blockSize2D_, blockSize2D_);
   dim3 dimGrid((int) ceil((numberOfDetectors_ / 2.0 + 1) / (float) blockSize2D_),
         (int) ceil(numberOfProjections_ / (float) blockSize2D_));
   auto filterFunction_d = glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(filter_.size());
   CHECK(cudaMemcpy(filterFunction_d.get(), filter_.data(), sizeof(float)*filter_.size(), cudaMemcpyHostToDevice));
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
            (numberOfDetectors_ / 2) + 1, numberOfProjections_, sinoFreq.get(), filterFunction_d.get());

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

auto Filter::initCuFFT(const int deviceID) -> void {

   CHECK(cudaSetDevice(deviceID));

   cudaStream_t stream;
   cufftHandle planFwd, planInv;

   CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 3));
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
      filter_.push_back(filterValue/(float)numberOfDetectors_);
   }
}

auto Filter::readConfig(const read_json& config_reader) -> bool {
	std::string filterType;
	try {
	   numberOfProjections_ = config_reader.get_value<int>("number_of_par_proj");
	   numberOfDetectors_ = config_reader.get_value<int>("number_of_par_det");
	   numberOfPixels_ = config_reader.get_value<int>("number_of_pixels");
	   blockSize2D_ = config_reader.get_value<int>("blocksize_2d_filter");
	   filterType = config_reader.get_value<std::string>("filter_type");
	   cutoffFraction_ = config_reader.get_value<float>("cut_off_fraction");
	}catch (const boost::property_tree::ptree_error& e) {
		BOOST_LOG_TRIVIAL(error) << "risa::cuda::Filter: Failed to read config: " << e.what();
	   return EXIT_FAILURE;
	}
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

//!<  CUDA Kernel that weights all the projections with the filter function
/**
 *    The variable i represents the detector index in the parallel beam sinogram,
 *    the variable j represents the projection in the parallel beam sinogram.
 *
 *    @param[in]  x  the number of detectors in the parallel beam sinogram
 *    @param[in]  y  the number of projections in the parallel beam sinogram
 *    @param[in,out] data  the inverse transformed parallel ray sinogram
 *    @param[in]  filter   pointer to the precomputed filter function
 */
__global__ void applyFilter(const int x, const int y, cufftComplex *data , const float* const __restrict__ filter) {
   const int j = blockIdx.y * blockDim.y + threadIdx.y;
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < x && j < y) {
      //cufft performs an unnormalized transformation ifft(fft(A))=length(A)*A
      //->normalization needs to be performed
      //const float filterVal = filter_d[i] * normalization;
      data[i + j * x].x *= filter[i];
      data[i + j * x].y *= filter[i];
   }
}

}
}
