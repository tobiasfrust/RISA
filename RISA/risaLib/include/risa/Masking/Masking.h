/*
 * Copyright 2016
 *
 * CropImage.h
 *
 *  Created on: 31.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef MASKING_H_
#define MASKING_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>

namespace risa {
namespace cuda {

/**
 * This class represents a masking stage. It multiplies the reconstructed image with
 * a precomputed mask in a CUDA kernel, to hide irrelevant areas.
 */

class Masking {

public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;

public:

   //!   Initializes everything, that needs to be done only once
   /**
    *
    *    Runs as many processor-thread as CUDA devices are available in the system.
    *
    *    @param[in]  configFile  path to configuration file
    */
   Masking(const std::string& configFile);

   //!   Destroys everything that is not destroyed automatically
   /**
    *   Destroys the cudaStreams.
    */
   ~Masking();

   //! Pushes the filtered parallel beam sinogram to the processor-threads
   /**
    *    @param[in]  inp   input data that arrived from previous stage
    */
   auto process(input_type&& img) -> void;

   //! Takes one image from the output queue #results_ and transfers it to the neighbored stage.
   /**
    *    @return  the oldest reconstructed image in the output queue #results_
    */
   auto wait() -> output_type;

private:

   std::map<int, ddrf::Queue<input_type>> imgs_;   //!<  one separate input queue for each available CUDA device
   ddrf::Queue<output_type> results_;              //!<  the output queue in which the processed sinograms are stored

   std::map<int, std::thread> processorThreads_;   //!<  stores the processor()-threads
   std::map<int, cudaStream_t> streams_;           //!<  stores the cudaStreams that are created once

   //! main data processing routine executed in its own thread for each CUDA device, that performs the data processing of this stage
   /**
    * This method takes one reconstruced image from the queue. It calls the masking
    * CUDA kernel in its own stream. After the multiplication of the mask with the image, the
    * result is pushed into the output queue
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    */
   auto processor(int deviceID) -> void;

   int numberOfDevices_;                           //!<  the number of available CUDA devices in the system

   int numberOfPixels_;                            //!<  the number of pixels in the reconstruction grid in one dimension

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

#endif /* CROPIMAGE_H_ */
