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

class Fan2Para {
public:
   using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
   using deviceManagerType = ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>;

public:
   Fan2Para(const std::string& configFile);

   ~Fan2Para();

   auto process(input_type&& fanSinogram) -> void;
   auto wait() -> output_type;

protected:

private:
   std::map<int, ddrf::Queue<input_type>> fanSinograms_;
   ddrf::Queue<output_type> results_;

   std::map<int, std::thread> processorThreads_;
   std::map<int, cudaStream_t> streams_;
   std::vector<unsigned int> memoryPoolIdxs_;
   int numberOfDevices_;

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
   int blockSize2D_, blockSize1D_;

   int memPoolSize_;

   auto processor(const int deviceID) -> void;
   auto initFan2Para() -> void;
   auto computeFan2ParaTransp() -> void;
   auto computeAngles(int i, int j, unsigned int ind, int k, float L,
         float kappa) -> void;
   auto transferToDevice(unsigned int deviceID) -> void;
   auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* FAN2PARA_H_ */
