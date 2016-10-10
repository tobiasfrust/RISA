/*
 * Copyright 2016
 *
 * Attenuation.h
 *
 *  Created on: 02.06.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef ATTENUATION_H_
#define ATTENUATION_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>
#include <string>

namespace risa {
namespace cuda {

//!   CUDA kernel to compute the attenuation coefficients
/**
 * This CUDA kernel computes the attenuation coefficient for the fan to parallel beam sinogram.
 * Furthermore, it multiplies the resulting values with a precomputed mask to hide unrelevant
 * areas, that are previously known by the geometry of the measurement system.
 *
 * @param[in]  sinogram_in the pointer to the raw data sinogram of size numberOfDetectors*numberOfProjections
 * @param[in]  mask  the pointer to the mask values, that is multiplied with the attenuation coefficient
 * @param[out] sinogram_out   pointer to the fan to parallel beam sinogram
 * @param[in]  avgReference   pointer to the averaged reference measurement on device
 * @param[in]  avgDark  pointer to the averaged dark measurement on device
 * @param[in]  temp
 * @param[in]  numberOfDetectors the number of detectors in the fan beam sinogram
 * @param[in]  numberOfProjections  the number of projections in the fan beam sinogram
 * @param[in]  planeId  the id of the sinogram's plane
 *
 */
__global__ void computeAttenuation(
      const unsigned short* __restrict__ sinogram_in,
      const float* __restrict__ mask, float* __restrict__ sinogram_out,
      const float* __restrict__ avgReference, const float* __restrict__ avgDark,
      const float temp, const int numberOfDetectors,
      const int numberOfProjections, const int planeId);

//! This stage computes the attenuation coefficients in the fan beam sinogram
/**
 * This class represents the attenuation stage. It computes the attenuation data
 * on the GPU device using the CUDA language. Multi GPU usage is possible.
 */
class Attenuation {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   //!< The input data type that needs to fit the output type of the previous stage
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   //!< The output data type that needs to fit the input type of the following stage
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>;
public:

   //!   Initializes everything, that needs to be done only once
   /**
    *
    *    Runs as many processor-thread as CUDA devices are available in the system. Allocates memory using the
    *    MemoryPool for all CUDA devices.
    *
    *    @param[in]  configFile  path to configuration file
    */
   Attenuation(const std::string& configFile);

   //!   Destroys everything that is not destroyed automatically
   /**
    *    Tells MemoryPool to free the allocated memory.
    *    Destroys the cudaStreams.
    */
   ~Attenuation();

   //! Pushes the sinogram to the processor-threads
   /**
    *    @param[in]  img   input data that arrived from previous stage
    */
   auto process(input_type&& img) -> void;

   //! Takes one sinogram from the output queue #results_ and transfers it to the neighbored stage.
   /**
    *    @return  the oldest sinogram in the output queue #results_
    */
   auto wait() -> output_type;

private:

   std::map<int, ddrf::Queue<input_type>> sinograms_; //!<  one separate input queue for each available CUDA device
   ddrf::Queue<output_type> results_;                 //!<  the output queue in which the processed sinograms are stored

   std::map<int, std::thread> processorThreads_;      //!<  stores the processor()-threads
   std::map<int, cudaStream_t> streams_;              //!<  stores the cudaStreams that are created once
   std::map<int, unsigned int> memoryPoolIdxs_;       //!<  stores the indeces received when regisitering in MemoryPool

   //! main data processing routine executed in its own thread for each CUDA device, that performs the data processing of this stage
   /**
    * This method takes one sinogram from the queue. It calls the attenuation
    * CUDA kernel in its own stream. After the computation of the attenuation data, the
    * fan beam sinogram is pushed into the output queue
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    */
   auto processor(int deviceID) -> void;

   //!   Capsules the average computation of dark and reference measurement.
   /**
    *    Fills the host vectors #avgDark_ and #avgReference_ with the computed data
    *    from the input files.
    */
   auto init() -> void;

   //!   computes the average values from the given input files
   /**
    *    @param[in]  values   vector, containing the data read from the input files
    *    @param[out] average  vector, containing the averaged data
    */
   template <typename T>
   auto computeAverage(const std::vector<T>& values, std::vector<float>&average) -> void;

   template <typename T>
   auto computeDarkAverage(const std::vector<T>& values, std::vector<float>& average) -> void;

   //!
   /**
    *
    */
   template <typename T>
   auto readDarkInputFiles(std::string& file, std::vector<T>& values) -> void;

   //!
   /**
    *
    */
   template <typename T>
   auto readInput(std::string& path, std::vector<T>& values, const int numberOfFrames) -> void;

   //!   Computes a mask to hide the unrelevant areas in the fan beam sinogram.
   /**
    *    Due to the special geometry of ROFEX (e.g. limited angle) there are areas
    *    where it is known from a priori knowledge that all values need to be zero.
    *    This mask is multiplied with the fan beam sinogramm after the attenuation
    *    computation.
    *
    *    @param[out] mask  contains the values of the mask
    */
   template <typename T>
   auto relevantAreaMask(std::vector<T>& mask) -> void;

   int numberOfDevices_;         //!<  the number of available CUDA devices in the system

   unsigned int chunkSize_{500u}; //!<  defines how much input data is loaded from reference and dark input at once

   //configuration values
   int numberOfDetectorModules_; //!<  the number of detector modules
   int numberOfDetectors_;       //!<  the number of detectors in the fan beam sinogram
   int numberOfProjections_;     //!<  the number of projections in the fan beam sinogram
   int numberOfPlanes_;          //!<  the number of detector planes
   int numberOfDarkFrames_;      //!<  the number of frames in the dark measurement
   int numberOfRefFrames_;       //!<  the number of frames in the reference measurement
   std::string pathDark_;        //!<  file path to dark measurement data
   std::string pathReference_;   //!<  file path to reference measurement data

   //parameters for mask generation
   double sourceOffset_;         //!<  source offset in the fan beam sinogram
   double lowerLimOffset_;       //!<  lower offset, which is masked
   double upperLimOffset_;       //!<  upper offset, which is masked
   unsigned int xa_;             
   unsigned int xb_;
   unsigned int xc_;
   unsigned int xd_;
   unsigned int xe_;
   unsigned int xf_;

   double threshMin_;            //!<  minimum threshold for defect detector interpolation
   double threshMax_;            //!<  maximum threshold for defect detector interpolation

   //kernel execution coniguration
   int blockSize2D_;             //!<  2D block size of the attenuation kernel
   int memPoolSize_;             //!<  specifies, how many elements are allocated by memory pool

   //average values on host
   std::vector<float> avgDark_;        //!<  stores averaged dark measurement on host
   std::vector<float> avgReference_;   //!<  stores averaged reference measurement on host


   //!  Read configuration values from configuration file
   /**
    * All values needed for setting up the class are read from the config file
    * in this function.
    *
    * @param[in] configFile path to config file
    *
    * @retval  true  configuration options were read successfully
    * @retval  false configuration options could not be read successfully
    */
   auto readConfig(const std::string& configFile) -> bool;

};
}
}

#endif /* ATTENUATION_H_ */
