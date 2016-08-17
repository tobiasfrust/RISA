/*
 *  Copyright 2016
 *
 *  cuda_kernels_fan2para.h
 *
 *  Created on: 17.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#ifndef CUDA_KERNELS_FAN2PARA_H_
#define CUDA_KERNELS_FAN2PARA_H_

#include <risa/Fan2Para/Fan2Para.h>

#include <ddrf/cuda/Coordinates.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>

namespace risa {
namespace cuda {

template<typename T>
auto ellipse_kreis_uwe(T alpha, T DX, T DZ, T SourceRingDiam) -> T {

   //Hilfsvariablen
   T L, R, CA, eps, p1, p2, gamma, ae;

   L = sqrt(DX * DX + DZ * DZ);
   R = 0.5 * SourceRingDiam;
   CA = cos(alpha);

   eps = (L * L + R * DX * CA) / (L * sqrt(L * L + R * R + 2.0 * R * DX * CA));
   eps = acos(eps);

   p1 = (L * L - R * DX) / (L * sqrt(L * L + R * R - 2.0 * R * DX));
   p2 = (L * L + R * DX) / (L * sqrt(L * L + R * R + 2.0 * R * DX));
   gamma = 0.5 * (acos(p1) - acos(p2));

   ae = (eps * CA + gamma)
         / sqrt(eps * eps + 2.0 * eps * gamma * CA + gamma * gamma);

   if (alpha <= M_PI)
      return acos(ae);
   else
      return 2.0 * M_PI - acos(ae);
}

__global__ void interpolation(int k, const float* __restrict__ SinFan_data,
      float* __restrict__ SinPar_data, const float* __restrict__ Gamma,
      const float* __restrict__ Teta, const float* __restrict__ alpha_kreis,
      const float* __restrict__ s, const int* __restrict__ teta_nach_ray_1,
      const int* __restrict__ teta_nach_ray_2,
      const int* __restrict__ teta_vor_ray_1,
      const int* __restrict__ teta_vor_ray_2,
      const int* __restrict__ gamma_vor_ray_1,
      const int* __restrict__ gamma_vor_ray_2,
      const int* __restrict__ gamma_nach_ray_1,
      const int* __restrict__ gamma_nach_ray_2,
      const float* __restrict__ teta_ziel_ray_1,
      const float* __restrict__ teta_ziel_ray_2,
      const float* __restrict__ gamma_ziel_ray_1,
      const float* __restrict__ gamma_ziel_ray_2, const int* __restrict__ ray_1,
      const int* __restrict__ ray_2, const parameters* __restrict__ params) {

   int i = ddrf::cuda::getX();
   int j = ddrf::cuda::getY();
   //finish all threads, that operate outside the bounds of the data field
   if (i >= (*params).numberOfParallelDetectors_
         && j >= (*params).numberOfParallelProjections_)
      return;

   //Übergabe-Paramter
   float WZiel_1 = 0, WZiel_2 = 0, WZiel_end = 0, V1 = 0, V2 = 0, W1 = 0,
         W2 = 0, W3 = 0, W4 = 0;

   //k defines, which plane to use
   unsigned long long ind = j * (*params).numberOfParallelDetectors_ + i
         + (k * (*params).numberOfParallelProjections_
               * (*params).numberOfParallelDetectors_);

   float temp_1 = s[i] / (*params).rDetector_;

   if (temp_1 <= 1 || temp_1 >= -1) {

      if (ray_1[ind] == true) {
         //Interpolationspunkte nehmen für Fall 1
         W1 = SinFan_data[teta_vor_ray_1[ind] * (*params).numberOfFanDetectors_
               + gamma_vor_ray_1[ind]];
         W2 = SinFan_data[teta_vor_ray_1[ind] * (*params).numberOfFanDetectors_
               + gamma_nach_ray_1[ind]];
         W3 = SinFan_data[teta_nach_ray_1[ind] * (*params).numberOfFanDetectors_
               + gamma_vor_ray_1[ind]];
         W4 = SinFan_data[teta_nach_ray_1[ind] * (*params).numberOfFanDetectors_
               + gamma_nach_ray_1[ind]];

         //Interpolation durchführen für Fall 1
         V1 = W1
               + ((teta_ziel_ray_1[ind] - Teta[teta_vor_ray_1[ind]])
                     / (Teta[teta_nach_ray_1[ind]] - Teta[teta_vor_ray_1[ind]]))
                     * (W3 - W1);
         V2 = W2
               + ((teta_ziel_ray_1[ind] - Teta[teta_vor_ray_1[ind]])
                     / (Teta[teta_nach_ray_1[ind]] - Teta[teta_vor_ray_1[ind]]))
                     * (W4 - W2);
         WZiel_1 = V1
               + ((gamma_ziel_ray_1[ind] - Gamma[gamma_vor_ray_1[ind]])
                     / (Gamma[gamma_nach_ray_1[ind]]
                           - Gamma[gamma_vor_ray_1[ind]])) * (V2 - V1);
      }
      if (ray_2[ind] == true) {

         //Interpolationspunkte nehmen für Fall 2
         W1 = SinFan_data[teta_vor_ray_2[ind] * (*params).numberOfFanDetectors_
               + gamma_vor_ray_2[ind]];
         W2 = SinFan_data[teta_vor_ray_2[ind] * (*params).numberOfFanDetectors_
               + gamma_nach_ray_2[ind]];
         W3 = SinFan_data[teta_nach_ray_2[ind] * (*params).numberOfFanDetectors_
               + gamma_vor_ray_2[ind]];
         W4 = SinFan_data[teta_nach_ray_2[ind] * (*params).numberOfFanDetectors_
               + gamma_nach_ray_2[ind]];

         //Interpolation durchführen für Fall 2
         V1 = W1
               + ((teta_ziel_ray_2[ind] - Teta[teta_vor_ray_2[ind]])
                     / (Teta[teta_nach_ray_2[ind]] - Teta[teta_vor_ray_2[ind]]))
                     * (W3 - W1);
         V2 = W2
               + ((teta_ziel_ray_2[ind] - Teta[teta_vor_ray_2[ind]])
                     / (Teta[teta_nach_ray_2[ind]] - Teta[teta_vor_ray_2[ind]]))
                     * (W4 - W2);
         WZiel_2 = V1
               + ((gamma_ziel_ray_2[ind] - Gamma[gamma_vor_ray_2[ind]])
                     / (Gamma[gamma_nach_ray_2[ind]]
                           - Gamma[gamma_vor_ray_2[ind]])) * (V2 - V1);
      }

      if (ray_1[ind] + ray_2[ind] > 0)

         WZiel_end = (float) ray_1[ind]
               / ((float) ray_1[ind] + (float) ray_2[ind]) * WZiel_1
               + (float) ray_2[ind] / ((float) ray_1[ind] + (float) ray_2[ind])
                     * WZiel_2;
   }
   //conversion from 360 to 180 degrees
   const int address = j * (*params).numberOfParallelDetectors_;
   const int parallelSize = (*params).numberOfParallelProjections_
         * (*params).numberOfParallelDetectors_;
   const int detectorSize_2 = (*params).numberOfParallelDetectors_ / 2;
   int mirrorOffset = 0;
   if (i < detectorSize_2)
      mirrorOffset = (*params).numberOfParallelDetectors_ - i - 1;
   else
      mirrorOffset = -(i % detectorSize_2) + detectorSize_2 - 1;
   if (j < ((*params).numberOfParallelProjections_ / 2))
      //first half of the parallel sinogram
      atomicAdd(&SinPar_data[address + i], WZiel_end * 0.5);
   else
      //second half of the parallel sinogram
      atomicAdd(&SinPar_data[address - parallelSize / 2 + mirrorOffset],
            WZiel_end * 0.5);
}

template<typename T>
__global__ void setValue(T* data, T value, int size) {
   auto id = ddrf::cuda::getX();
   if (id >= size)
      return;
   data[id] = value;
}

}
}

#endif //CUDA_KERNELS_FAN2PARA_H_
