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

#ifndef FILTER_H_
#define FILTER_H_

#include "../ConfigReader/read_json.hpp"

#include <glados/Image.h>
#include <glados/cuda/DeviceMemoryManager.h>
#include <glados/Queue.h>
#include <glados/cuda/Memory.h>

#include <thrust/device_vector.h>

#include <cufft.h>

#include <map>
#include <thread>

namespace risa {
namespace cuda {

   namespace detail{
      /**
      *  This enum represents the filter type
      *  to be used during the filtering
      */
      enum FilterType: short {
         ramp,
         sheppLogan,
         cosine,
         hamming,
         hanning
      };
   }

//! This stage filters the projections in the parallel beam sinogram with a precomputed filter function.
class Filter {
public:
	using input_type = glados::Image<glados::cuda::DeviceMemoryManager<float, glados::cuda::async_copy_policy>>;
   //!< The input data type that needs to fit the output type of the previous stage
   using output_type = glados::Image<glados::cuda::DeviceMemoryManager<float, glados::cuda::async_copy_policy>>;
   //!< The output data type that needs to fit the input type of the following stage

public:

   //!   Initializes everything, that needs to be done only once
   /**
    *
    *    Runs as many processor-thread as CUDA devices are available in the system.
    *
    *    @param[in]  configFile  path to configuration file
    */
	Filter(const std::string& config_file);

   //!   Destroys everything that is not destroyed automatically
   /**
    *    Destroys the cudaStreams.
    */
	~Filter();

   //! Pushes the filtered parallel beam sinogram to the processor-threads
   /**
    *    @param[in]  inp   input data that arrived from previous stage
    */
	auto process(input_type&& sinogram) -> void;

   //! Takes one image from the output queue #results_ and transfers it to the neighbored stage.
   /**
    *    @return  the oldest reconstructed image in the output queue #results_
    */
	auto wait() -> output_type;

private:
	std::map<int, glados::Queue<input_type>> sinograms_;   //!<  one separate input queue for each available CUDA device
	glados::Queue<output_type> results_;                   //!<  the output queue in which the processed sinograms are stored

	std::map<int, std::thread> processorThreads_;        //!<  stores the processor()-threads

	int numberOfProjections_;                            //!<  the number of projections in the parallel beam sinogramm over 180 degrees
	int numberOfDetectors_;                              //!<  the number of detectors in the parallel beam sinogramm over 180 degrees
	int numberOfPixels_;                                 //!<  the number of pixels in the reconstruction grid in one dimension

	detail::FilterType filterType_;                      //!<  the filter type that shall be used; standarf filter type is the ramp filter.
	float cutoffFraction_;                               //!<  the fraction at which the filter function is cropped and set to zero.

	int numberOfDevices_;                       //!<  the number of available CUDA devices in the system

   //kernel execution coniguration
   int blockSize2D_;                           //!<  the block size of the filter kernel

	std::map<int, cufftHandle> plansFwd_;       //!<  the forward plans for the cuFFT forward transformation; for each device one;
	std::map<int, cufftHandle> plansInv_;       //!<  the inverse plans for the cuFFT inverse tranformation;  for each device one;

	std::map<int, cudaStream_t> streams_;       //!<  stores the cudaStreams that are created once

	std::vector<float> filter_;                 //!<  stores the values of the filter function

   //! main data processing routine executed in its own thread for each CUDA device, that performs the data processing of this stage
   /**
    * This method takes one sinogram from the queue. It calls the desired filter
    * CUDA kernel in its own stream. After the computation of the filtered projections the
    * filtered parallel sinogram is pushed into the output queue
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    */
	auto processor(const int deviceID) -> void;

   //!   initializes the cuFFT and creates the forward and inverse plans once for each deviceID
   /**
    *
    * @param[in]  deviceID the ID of the device that shall be initialized for cuFFT
    */
	auto initCuFFT(const int deviceID) -> void;

   //!  Read configuration values from configuration file
   /**
    * All values needed for setting up the class are read from the config file
    * in this function. If an invalid filter function is requested, the ramp filter is used.
    *
    * @param[in] configFile path to config file
    *
    * @retval  true  configuration options were read successfully
    * @retval  false configuration options could not be read successfully
    */
	auto readConfig(const read_json& config_reader) -> bool;

   //!<  This function computes the requested filter function once on the host
	auto designFilter() -> void;
};
}
}

#endif /* FILTER_H_ */
