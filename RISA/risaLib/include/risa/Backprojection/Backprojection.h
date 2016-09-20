/*
 * Copyright 2016
 *
 * Backprojection.h
 *
 *  Created on: 26.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef BACKPROJECTION_H_
#define BACKPROJECTION_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>

#include <thrust/device_vector.h>

#include <map>
#include <thread>
#include <array>

namespace risa {
namespace cuda {


   namespace detail{
      /**
      *  This enum represents the type of interpolation
      *  to be used during the back projection
      */
      enum InterpolationType: short {
         neareastNeighbor,
         linear
      };
   }

   //! This function performs the back projection operation with linear interpolation
   /**
    * With a pixel-driven back projection approach, this CUDA kernel spans number of pixels
    * times number of pixels, in the reconstruction grid, threads. Starting at the pixel center,
    * each thread follows the ray path and computes the intersection with the detector pixels.
    * The intersection does not line up, and thus, a linear interpolation is performed.
    *
    * @param[in]  sinogram             linearized sinogram data. Each projection is stored linearly after each other
    * @param[out] image                the reconstruction grid, in which the reconstructed image is stored
    * @param[in]  numberOfPixels       the number of pixels in the reconstruction grid in one dimension
    * @param[in]  numberOfProjections  the number of projections in the parallel beam sinogram over 180 degrees
    * @param[in]  numberOfDetectors    the number of detectors in the parallel beam sinogram
    */
   __global__ void backProjectLinear(const float* const __restrict__ sinogram,
         float* __restrict__ image, const int numberOfPixels,
         const int numberOfProjections, const int numberOfDetectors);


   //! This function performs the back projection with nearest neighbor interpolation
   /**
    * With a pixel-driven back projection approach, this CUDA kernel spans number of pixels
    * times number of pixels, in the reconstruction grid, threads. Starting at the pixel center,
    * each thread follows the ray path and computes the intersection with the detector pixels.
    * The intersection does not line up, and thus, a nearest neigbor interpolation is performed.
    *
    * @param[in]  sinogram             linearized sinogram data. Each projection is stored linearly after each other
    * @param[out] image                the reconstruction grid, in which the reconstructed image is stored
    * @param[in]  numberOfPixels       the number of pixels in the reconstruction grid in one dimension
    * @param[in]  numberOfProjections  the number of projections in the parallel beam sinogram over 180 degrees
    * @param[in]  numberOfDetectors    the number of detectors in the parallel beam sinogram
    */
   __global__ void backProjectNearest(const float* const __restrict__ sinogram,
         float* __restrict__ image, const int numberOfPixels,
         const int numberOfProjections, const int numberOfDetectors);


   //!   This stage back projects a parallel beam sinogram and returns the reconstructed image.
   /**
    * This class represents the back projection stage. It computes the back projection on the GPU
    * using the CUDA language. Multi GPU usage is supported.
    */
class Backprojection {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
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
   Backprojection(const std::string& configFile);

   //!   Destroys everything that is not destroyed automatically
   /**
    *    Tells MemoryPool to free the allocated memory.
    *    Destroys the cudaStreams.
    */
   ~Backprojection();

   //! Pushes the filtered parallel beam sinogram to the processor-threads
   /**
    *    @param[in]  inp   input data that arrived from previous stage
    */
   auto process(input_type&& inp) -> void;

   //! Takes one image from the output queue #results_ and transfers it to the neighbored stage.
   /**
    *    @return  the oldest reconstructed image in the output queue #results_
    */
   auto wait() -> output_type;

protected:

private:
   std::map<int, ddrf::Queue<input_type>> sinograms_; //!<  one separate input queue for each available CUDA device
   ddrf::Queue<output_type> results_;                 //!<  the output queue in which the processed sinograms are stored

   std::map<int, std::thread> processorThreads_;      //!<  stores the processor()-threads
   std::map<int, cudaStream_t> streams_;              //!<  stores the cudaStreams that are created once
   std::vector<unsigned int> memoryPoolIdxs_;         //!<  stores the indeces received when regisitering in MemoryPool

   int numberOfProjections_;                          //!<  the number of projections in the parallel beam sinogramm over 180 degrees
   int numberOfDetectors_;                            //!<  the number of detectors in the parallel beam sinogramm
   int numberOfPixels_;                               //!<  the number of pixels in the reconstruction grid in one dimension
   float rotationOffset_;                             //!<  the rotation of the reconstructed image
   float backProjectionAngleTotal_;

   int numberOfDevices_;                              //!<  the number of available CUDA devices in the system
   int numberOfStreams_;                              //!<  the number of different CUDA streams to be used

   detail::InterpolationType interpolationType_;      //!<  the interpolation type that shall be used

   //kernel execution coniguration
   int blockSize2D_;                                  //!<  2D block size of the back projection kernel
   int memPoolSize_;                                  //!<  specifies, how many elements are allocated by memory pool

   std::vector<int> lastStreams_;                     //!<  stores, which stream was used last

   bool useTextureMemory_;

   //! main data processing routine executed in its own thread for each CUDA device, that performs the data processing of this stage
   /**
    * This method takes one sinogram from the queue. It calls the desired back projection
    * CUDA kernel in its own stream. After the computation of the back projection, the
    * reconstructed image is pushed into the output queue
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    * @param[in]  streamID specifies on which CUDA stream to execute the device functions
    */
   auto processor(const int deviceID, const int streamID) -> void;

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

#endif /* BACKPROJECTION_H_ */
