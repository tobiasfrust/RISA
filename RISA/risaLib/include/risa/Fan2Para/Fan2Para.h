/*
 * Copyright 2016
 *
 * Fan2Para.h
 *
 *  Created on: 14.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef FAN2PARA_H_
#define FAN2PARA_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thrust/device_vector.h>

#include <map>
#include <thread>
#include <array>

namespace risa {
namespace cuda {

//! collects all parameters that are needed in the fan to parallel beam interpolation kernel
struct parameters {
   int numberOfPlanes_;
   int numberOfFanDetectors_;
   int numberOfFanProjections_;
   int numberOfParallelProjections_;
   int numberOfParallelDetectors_;
   float sourceOffset_;
   float detectorDiameter_;
   float rDetector_;
   float imageCenterX_;
   float imageCenterY_;
   float imageWidth_;
};

//! collects all precomputed hash table values
struct hashTable {
   float *Gamma;
   float *Teta;
   float *alpha_kreis;
   float *s;
   int *teta_nach_ray_1;
   int *teta_nach_ray_2;
   int *teta_vor_ray_1;
   int *teta_vor_ray_2;
   int *gamma_vor_ray_1;
   int *gamma_vor_ray_2;
   int *gamma_nach_ray_1;
   int *gamma_nach_ray_2;
   float *teta_ziel_ray_1;
   float *teta_ziel_ray_2;
   float *gamma_ziel_ray_1;
   float *gamma_ziel_ray_2;
   int *ray_1;
   int *ray_2;

};

//!   This stage performs the fan to parallel beam rebinning.
/**
 * This class represents the fan to parallel beam rebinning stage. It computes a hash table once at program
 * initialization. The fan to parallel beam interpolation is performed using a CUDA kernel.
 */
class Fan2Para {
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
   Fan2Para(const std::string& configFile);

   //!   Destroys everything that is not destroyed automatically
   /**
    *    Tells MemoryPool to free the allocated memory.
    *    Destroys the cudaStreams.
    */
   ~Fan2Para();

   //! Pushes the filtered parallel beam sinogram to the processor-threads
   /**
    *    @param[in]  inp   input data that arrived from previous stage
    */
   auto process(input_type&& fanSinogram) -> void;

   //! Takes one image from the output queue #results_ and transfers it to the neighbored stage.
   /**
    *    @return  the oldest reconstructed image in the output queue #results_
    */
   auto wait() -> output_type;

protected:

private:
   std::map<int, ddrf::Queue<input_type>> fanSinograms_; //!<  one separate input queue for each available CUDA device
   ddrf::Queue<output_type> results_;                    //!<  the output queue in which the processed sinograms are stored

   std::map<int, std::thread> processorThreads_;         //!<  stores the processor()-threads
   std::map<int, cudaStream_t> streams_;                 //!<  stores the cudaStreams that are created once
   std::vector<unsigned int> memoryPoolIdxs_;            //!<  stores the indeces received when regisitering in MemoryPool
   int numberOfDevices_;                                 //!<  the number of available CUDA devices

   //configuration parameters
   parameters params_;
   std::array<float, 2> sourceDiam_;
   std::array<float, 2> deltaX_;
   std::array<float, 2> deltaZ_;
   std::array<float, 2> sourceAngle_;
   std::array<float, 2> rTarget_;
   std::array<char, 2> detectorInter_;

   //Hash Table on device
   std::map<int, ddrf::cuda::device_ptr<float, ddrf::cuda::async_copy_policy>>
         theta_d_, gamma_d_, s_d_, alphaCircle_d_;
   std::map<int, ddrf::cuda::device_ptr<int, ddrf::cuda::async_copy_policy>>
         thetaAfterRay1_d_, thetaAfterRay2_d_, thetaBeforeRay1_d_,
         thetaBeforeRay2_d_, gammaAfterRay1_d_, gammaAfterRay2_d_,
         gammaBeforeRay1_d_, gammaBeforeRay2_d_;
   std::map<int, ddrf::cuda::device_ptr<float, ddrf::cuda::async_copy_policy>>
         gammaGoalRay1_d_, gammaGoalRay2_d_, thetaGoalRay1_d_, thetaGoalRay2_d_;
   std::map<int, ddrf::cuda::device_ptr<int, ddrf::cuda::async_copy_policy>>
         ray1_d_, ray2_d_;

   //Hash Table on host
   std::vector<float> theta_, gamma_, s_, alphaCircle_;
   std::vector<int> thetaAfterRay1_, thetaAfterRay2_, thetaBeforeRay1_,
         thetaBeforeRay2_, gammaAfterRay1_, gammaAfterRay2_, gammaBeforeRay1_,
         gammaBeforeRay2_;
   std::vector<float> gammaGoalRay1_, gammaGoalRay2_, thetaGoalRay1_,
         thetaGoalRay2_;
   std::vector<int> ray1_, ray2_;

   //kernel execution coniguration
   int blockSize2D_;    //!<  the block size of the fan to parallel beam kernel
   int blockSize1D_;    //!<  the block size of the set to specific value kernel

   int memPoolSize_;    //!<  the number of elements the memory pool allocates

   //! main data processing routine executed in its own thread for each CUDA device, that performs the data processing of this stage
   /**
    * This method takes one sinogram from the queue. It calls the fan to parallel beam interpolation
    * CUDA kernel in its own stream. After the computation of the fan to parallel beam rebinning, the
    * reconstructed image is pushed into the output queue
    *
    * @param[in]  deviceID specifies on which CUDA device to execute the device functions
    * @param[in]  streamID specifies on which CUDA stream to execute the device functions
    */
   auto processor(const int deviceID) -> void;

   //!   The main function for computing the hash table for the fan to parallel beam rebinning process
   auto computeFan2ParaTransp() -> void;

   auto computeAngles(int i, int j, unsigned int ind, int k, float L,
         float kappa) -> void;

   //! Transfers the hash table from host to the specified CUDA device.
   /**
    * @param[in]  deviceID specifies on which CUDA device to transfer the hash table
    */
   auto transferToDevice(unsigned int deviceID) -> void;

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

#endif /* FAN2PARA_H_ */
