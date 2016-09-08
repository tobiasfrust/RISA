/*
 * Copyright 2016
 *
 * H2D.h
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef H2D_H_
#define H2D_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include "../Basics/performance.h"

#include <atomic>
#include <thread>
#include <vector>
#include <map>
#include <mutex>

namespace risa {
namespace cuda {

/**
 *    This class transfers the data to be processed from host to device.
 *    Furthermore, it performs the scheduling between the available devices.
 *    Scheduling is done statically, so far.
 */
class H2D {
public:
   //!< The input data type that needs to fit the output type of the previous stage
   using input_type = ddrf::Image<ddrf::cuda::HostMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   //!< The output data type that needs to fit the input type of the following stage
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

public:

   //!   Initializes everything, that needs to be done only once
   /**
    *
    *    Runs as many processor-thread as CUDA devices are available in the system. Allocates memory using the
    *    MemoryPool for all CUDA devices.
    *
    *    @param[in]  configFile  path to configuration file
    */
   H2D(const std::string& configFile);

   //!   Destroys everything that is not destroyed automatically
   /**
    *    Tells MemoryPool to free the allocated memory.
    *    Destroys the cudaStreams.
    */
   ~H2D();

   //! Pushes the sinogram to the processor-threads
   /**
    * The scheduling for multi-GPU usage is done in this function.
    *
    * @param[in]  sinogram input data that arrived from previous stage
    */
   auto process(input_type&& sinogram) -> void;

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
    * This method takes one sinogram from the input queue #sinograms_. The sinogram is transfered to the device
    * using the asynchronous cudaMemcpyAsync()-operation. The resulting device strucutre is pushed back into
    * the output queue #results_.
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    */
   auto processor(int deviceID) -> void;

   double worstCaseTime_;     //!<  stores the worst case time between the arrival of two following sinograms
   double bestCaseTime_;      //!<  stores the besst case time between the arrival of two following sinograms
   Timer tmr_;                //!<  used to measure the timings

   std::size_t lastIndex_;    //!<  stores the index of the last sinogram. Used to analyze which percentage of the arrived sinograms could be reconstructed
   std::size_t lostSinos_;    //!<  stores the number of sinograms that could not be reconstructed

   std::size_t count_ { 0 };  //!<  counts the total number of reconstructed sinograms

   int lastDevice_;           //!<  stores, to which device the last arrived sinogram was sent

   int numberOfDevices_;      //!<  the number of available CUDA devices in the system

   int numberOfDetectors_;    //!<  the number of detectors in the fan beam sinogram
   int numberOfProjections_;  //!<  the number of projections in the fan beam sinogram

   int memPoolSize_;          //!< specifies, how many elements are allocated by memory pool

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

#endif /* D2H_H_ */
