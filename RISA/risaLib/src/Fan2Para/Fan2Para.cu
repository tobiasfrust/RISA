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

#include "cuda_kernels_fan2para.h"

#include <risa/Fan2Para/Fan2Para.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <glados/MemoryPool.h>
#include <glados/cuda/Check.h>
#include <glados/cuda/Launch.h>

#include <boost/log/trivial.hpp>

#include <nvToolsExt.h>

#include <exception>
#include <vector>
#include <pthread.h>

namespace risa {
namespace cuda {

Fan2Para::Fan2Para(const std::string& configFile) {
   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::Fan2Para: Configuration file could not be loaded successfully. Please check!");
   }

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   auto dataSetSize = params_.numberOfParallelDetectors_
         * params_.numberOfParallelProjections_ * params_.numberOfPlanes_;

   //allocate memory for hash table on each device
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      theta_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  params_.numberOfFanProjections_));
      gamma_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  params_.numberOfFanDetectors_));
      s_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  params_.numberOfParallelDetectors_));
      alphaCircle_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  params_.numberOfParallelProjections_));
      thetaAfterRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      thetaAfterRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      thetaBeforeRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      thetaBeforeRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaAfterRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaAfterRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaBeforeRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaBeforeRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaGoalRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  dataSetSize));
      gammaGoalRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  dataSetSize));
      thetaGoalRay1_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  dataSetSize));
      thetaGoalRay2_d_[i] = std::move(
            glados::cuda::make_device_ptr<float, glados::cuda::async_copy_policy>(
                  dataSetSize));
      ray1_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
      ray2_d_[i] = std::move(
            glados::cuda::make_device_ptr<int, glados::cuda::async_copy_policy>(
                  dataSetSize));
   }

   computeFan2ParaTransp();

   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      //custom streams are necessary, because profiling with nvprof seems to be
      //not possible with -default-stream per-thread option
      cudaStream_t stream;
      CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 4));
      streams_[i] = stream;
   }

   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      transferToDevice(i);
   }

   //wait for all streams to be finished
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(streams_[i]));
   }

   //allocate memory on all available devices
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      memoryPoolIdxs_.push_back(
            glados::MemoryPool<deviceManagerType>::instance()->registerStage(memPoolSize_,
                  params_.numberOfParallelProjections_
                        * params_.numberOfParallelDetectors_/2.0));
   }

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &Fan2Para::processor, this, i };
   }
}

Fan2Para::~Fan2Para() {
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      glados::MemoryPool<deviceManagerType>::instance()->freeMemory(
            memoryPoolIdxs_[i]);
      CHECK(cudaStreamDestroy(streams_[i]));
   }
}

auto Fan2Para::process(input_type&& sinogram) -> void {
   if (sinogram.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "Fan2Para: Image arrived with Index: " << sinogram.index() << "to device " << sinogram.device();
      fanSinograms_[sinogram.device()].push(std::move(sinogram));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Fan2Para: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         fanSinograms_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }
      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Fan2Para: Finished.";
   }
}

auto Fan2Para::wait() -> output_type {
   return results_.take();
}

/**
 * The processor()-Method takes one sinogram from the queue.
 *
 */
auto Fan2Para::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "Fan2Para");
   CHECK(cudaSetDevice(deviceID));
   dim3 blocks2D(blockSize2D_, blockSize2D_);
   dim3 grids2D(
         std::ceil(params_.numberOfParallelDetectors_ / (float) blockSize2D_),
         std::ceil(
               params_.numberOfParallelProjections_ / (float) blockSize2D_));
   dim3 blocks1D(blockSize1D_);
   dim3 grids1D(
         std::ceil(
               params_.numberOfParallelProjections_
                     * params_.numberOfParallelDetectors_
                     / (float) blockSize1D_));
   auto params_d = glados::cuda::make_device_ptr<parameters>(1);
   CHECK(
         cudaMemcpyAsync(params_d.get(), &params_, sizeof(parameters),
               cudaMemcpyHostToDevice, streams_[deviceID]));
   CHECK(cudaStreamSynchronize(streams_[deviceID]));
   CHECK(cudaFuncSetCacheConfig(interpolation, cudaFuncCachePreferL1));
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Fan2Para: Running Thread for Device " << deviceID;
   while (true) {
      auto sinogram = fanSinograms_[deviceID].take();
      if (!sinogram.valid())
         break;
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Fan2Para: Fan2Para of sinogram with Index " << sinogram.index();

      //copy image from device to host
      auto img = glados::MemoryPool<deviceManagerType>::instance()->requestMemory(
            memoryPoolIdxs_[deviceID]);

      setValue<float> <<<grids1D, blocks1D, 0, streams_[deviceID]>>>(
            img.container().get(), 0.0, img.size());
      CHECK(cudaPeekAtLastError());

      interpolation<<<grids2D, blocks2D, 0, streams_[deviceID]>>>(
            sinogram.plane(), sinogram.container().get(), img.container().get(),
            gamma_d_[deviceID].get(), theta_d_[deviceID].get(),
            alphaCircle_d_[deviceID].get(), s_d_[deviceID].get(),
            thetaAfterRay1_d_[deviceID].get(),
            thetaAfterRay2_d_[deviceID].get(),
            thetaBeforeRay1_d_[deviceID].get(),
            thetaBeforeRay2_d_[deviceID].get(),
            gammaBeforeRay1_d_[deviceID].get(),
            gammaBeforeRay2_d_[deviceID].get(),
            gammaAfterRay1_d_[deviceID].get(),
            gammaAfterRay2_d_[deviceID].get(), thetaGoalRay1_d_[deviceID].get(),
            thetaGoalRay2_d_[deviceID].get(), gammaGoalRay1_d_[deviceID].get(),
            gammaGoalRay2_d_[deviceID].get(), ray1_d_[deviceID].get(),
            ray2_d_[deviceID].get(), params_d.get());
      CHECK(cudaPeekAtLastError());
      img.setDevice(deviceID);
      img.setIdx(sinogram.index());
      img.setPlane(sinogram.plane());
      img.setStart(sinogram.start());
      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(img));
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Fan2Para: Fan2Para of sinogram with Index " << sinogram.index() << " finished.";
   }
}

auto Fan2Para::computeFan2ParaTransp() -> void {

   BOOST_LOG_TRIVIAL(info)<< "Computing Hash Table for conversion from fan to parallel beam.";

   auto dataSetSize = params_.numberOfParallelProjections_ * params_.numberOfParallelDetectors_
   * params_.numberOfPlanes_;

   //allocate memory on host
   theta_.resize(params_.numberOfFanProjections_);
   gamma_.resize(params_.numberOfFanDetectors_);
   s_.resize(params_.numberOfParallelDetectors_);
   alphaCircle_.resize(params_.numberOfParallelProjections_);

   thetaAfterRay1_.resize(dataSetSize);
   thetaAfterRay2_.resize(dataSetSize);
   thetaBeforeRay1_.resize(dataSetSize);
   thetaBeforeRay2_.resize(dataSetSize);
   gammaAfterRay1_.resize(dataSetSize);
   gammaAfterRay2_.resize(dataSetSize);
   gammaBeforeRay1_.resize(dataSetSize);
   gammaBeforeRay2_.resize(dataSetSize);
   gammaGoalRay1_.resize(dataSetSize);
   gammaGoalRay2_.resize(dataSetSize);
   thetaGoalRay1_.resize(dataSetSize);
   thetaGoalRay2_.resize(dataSetSize);
   ray1_.resize(dataSetSize);
   ray2_.resize(dataSetSize);

   // === Init values for Hash table
   // ===============================
   // == Theta        = Ortswinkel des Quellpunktes auf Target
   // == Gamma       = Ortswinkel des Detektorpixels
   // == s           = diskreter Abstand der Detektorpixel (Para)
   // == alpha_kreis    = Ortswinkel der Parallelstrahlquellen

   for (auto j = 0; j < params_.numberOfFanProjections_; j++) {
      theta_[j] = j * (360.0 / params_.numberOfFanProjections_) - params_.sourceOffset_;
      if (theta_[j] < 0.0)
      theta_[j] = theta_[j] + 360.0;
      theta_[j] = ((2 * M_PI) / 360.0) * theta_[j];
   }

   gamma_[0] = 0.0;
   for (auto j = 1; j < params_.numberOfFanDetectors_; j++) {
      gamma_[j] = j * ((360.0 / (float) params_.numberOfFanDetectors_));
      gamma_[j] = ((2 * M_PI) / 360.0) * gamma_[j];
   }

   for (auto j = 0; j < params_.numberOfParallelDetectors_; j++)
   s_[j] = ((-0.5) * params_.imageWidth_)
   + (((0.5 + j) * params_.imageWidth_) / (float) params_.numberOfParallelDetectors_);

   for (auto j = params_.numberOfParallelProjections_; j > 0; j--) {
      alphaCircle_[j] = j * (360.0 / (float) params_.numberOfParallelProjections_);
      alphaCircle_[j] = ((2 * M_PI) / 360.0) * alphaCircle_[j] + M_PI / 2;

      if (alphaCircle_[j] > 2 * M_PI)
      alphaCircle_[j] = (alphaCircle_[j]) - (2.0 * M_PI);
   }

   // === calculate Hash Table
   // =========================
   unsigned long long ind = 0;
   int i, j, k;
   float kappa = 0, L = 0;
   float tb = 0.0;

   // Abfrage
   if (params_.imageCenterY_ != 0) {
      L = sqrt(params_.imageCenterY_ * params_.imageCenterY_ + params_.imageCenterX_ * params_.imageCenterX_);

      if (params_.imageCenterY_ < 0)
      tb = 1.0;

      kappa = atan(params_.imageCenterX_ / params_.imageCenterY_) + tb * M_PI;
   } else if (params_.imageCenterX_ != 0) {
      L = sqrt(params_.imageCenterY_ * params_.imageCenterY_ + params_.imageCenterX_ * params_.imageCenterX_);

      if (params_.imageCenterX_ < 0)
      kappa = -M_PI / 2.;
      else
      kappa = M_PI / 2.;
   }

   unsigned int parallelSize = params_.numberOfParallelDetectors_
   * params_.numberOfParallelProjections_;
   float temp_1;

   for (k = 0; k < params_.numberOfPlanes_; k++) {
      for (j = 0; j < params_.numberOfParallelProjections_; j++) {
         for (i = 0; i < params_.numberOfParallelDetectors_; i++) {

            ind = j * params_.numberOfParallelDetectors_ + i + (k * parallelSize);
            temp_1 = (s_[i] - L * sin(alphaCircle_[j] - kappa)) / params_.rDetector_;

            //Prüfen, ob asin möglich
            if (temp_1 <= 1 || temp_1 >= -1)
               computeAngles(i, j, ind, k, L, kappa);
         }
      }
   }
}

auto Fan2Para::computeAngles(int i, int j, unsigned int ind, int k, float L,
      float kappa) -> void {

   //Übergabe-Parameter
   float epsilon = 0;

   //Hilfsvariable
   float dif_best = M_PI, dif = 0, best_x = 0, temp_1 = 0, temp_2 = 0;
   int x = 0;

   //Berechnungsvorschrift
   //Theta
   temp_1 = asin(((s_[i] - L * sin(alphaCircle_[j] - kappa)) / rTarget_[k])); //<-----Veränderung
   thetaGoalRay1_[ind] = alphaCircle_[j] - temp_1;

   if (thetaGoalRay1_[ind] < 0)
      thetaGoalRay1_[ind] = thetaGoalRay1_[ind] + 2.0 * M_PI;

   thetaGoalRay1_[ind] = ellipse_kreis_uwe(thetaGoalRay1_[ind], deltaX_[k],
         deltaZ_[k], 2 * rTarget_[k]);

   thetaGoalRay2_[ind] = alphaCircle_[j] + temp_1 - M_PI;
   if (thetaGoalRay2_[ind] < 0)
      thetaGoalRay2_[ind] = thetaGoalRay2_[ind] + 2.0 * M_PI;

   thetaGoalRay2_[ind] = ellipse_kreis_uwe(thetaGoalRay2_[ind], deltaX_[k],
         deltaZ_[k], 2 * rTarget_[k]);

   temp_1 = ((360.0 - sourceAngle_[k]) / 2.0) / 180.0 * M_PI;
   temp_2 = (360.0 - ((360.0 - sourceAngle_[k]) / 2.0)) / 180.0 * M_PI;
   if (thetaGoalRay1_[ind] > temp_1 && thetaGoalRay1_[ind] < temp_2)
      ray1_[ind] = 1;
   if (thetaGoalRay2_[ind] > temp_1 && thetaGoalRay2_[ind] < temp_2)
      ray2_[ind] = 1;

   epsilon = asin(
         ((s_[i] - L * sin(alphaCircle_[j] - kappa)) / params_.rDetector_)); //<-----Veränderung

   if (ray1_[ind]) {

      //Gamma
      gammaGoalRay1_[ind] = epsilon + alphaCircle_[j] - 1.5 * M_PI;
      if (gammaGoalRay1_[ind] < 0)
         gammaGoalRay1_[ind] = gammaGoalRay1_[ind] + 2.0 * M_PI;
      if (gammaGoalRay1_[ind] > 2 * M_PI)
         gammaGoalRay1_[ind] = gammaGoalRay1_[ind] - 2.0 * M_PI;

      //Vektor Teta nach Wert durchsuchen für Fall 1
      for (x = 0; x < params_.numberOfFanProjections_; x++) {
         if (thetaGoalRay1_[ind] <= theta_[x]) {
            dif = theta_[x] - thetaGoalRay1_[ind];
            if (dif < dif_best) {
               dif_best = dif;
               best_x = x;
            }
         }
      }

      if (best_x == 0) {
         thetaBeforeRay1_[ind] = params_.numberOfFanProjections_ - 1;
         thetaAfterRay1_[ind] = best_x;
      } else {
         thetaBeforeRay1_[ind] = best_x - 1;
         thetaAfterRay1_[ind] = best_x;
      }

      //Vektor Gamma nach Wert durchsuchen für Fall 1
      for (x = 0; x < params_.numberOfFanDetectors_; x++) {
         if (gammaGoalRay1_[ind] <= gamma_[x]) {
            if (x == 0)
               gammaBeforeRay1_[ind] = params_.numberOfFanDetectors_ - 1;
            else
               gammaBeforeRay1_[ind] = x - 1;
            gammaAfterRay1_[ind] = x;
            break;
         }
      }
      if (gammaGoalRay1_[ind] > gamma_[params_.numberOfFanDetectors_ - 1]) {
         gammaBeforeRay1_[ind] = params_.numberOfFanDetectors_ - 1;
         gammaAfterRay1_[ind] = 0;
      }
   }

   if (ray2_[ind]) {
      dif_best = M_PI;

      //Gamma für Fall 2
      gammaGoalRay2_[ind] = -epsilon + alphaCircle_[j] - (M_PI / 2.0);
      if (gammaGoalRay2_[ind] < 0)
         gammaGoalRay2_[ind] = gammaGoalRay2_[ind] + 2.0 * M_PI;

      //Vektor Teta nach Wert durchsuchen für Fall 2
      for (x = 0; x < params_.numberOfFanProjections_; x++) {
         if (thetaGoalRay2_[ind] <= theta_[x]) {
            dif = theta_[x] - thetaGoalRay2_[ind];
            if (dif < dif_best) {
               dif_best = dif;
               best_x = x;
            }
         }
      }
      if (best_x == 0) {
         thetaBeforeRay2_[ind] = params_.numberOfFanProjections_ - 1;
         thetaAfterRay2_[ind] = best_x;

      } else {
         thetaBeforeRay2_[ind] = best_x - 1;
         thetaAfterRay2_[ind] = best_x;
      }

      //Vektor Gamma nach Wert durchsuchen für Fall 2
      for (x = 0; x < params_.numberOfFanDetectors_; x++) {
         if (gammaGoalRay2_[ind] <= gamma_[x]) {
            if (x == 0)
               gammaBeforeRay2_[ind] = params_.numberOfFanDetectors_ - 1;
            else
               gammaBeforeRay2_[ind] = x - 1;
            gammaAfterRay2_[ind] = x;
            break;
         }
      }
      if (gammaGoalRay2_[ind] > gamma_[params_.numberOfFanDetectors_ - 1]) {
         gammaBeforeRay2_[ind] = params_.numberOfFanDetectors_ - 1;
         gammaAfterRay2_[ind] = 0;
      }
   }
}

auto Fan2Para::transferToDevice(unsigned int deviceID) -> void {
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&(theta_d_[deviceID][0])),
               thrust::raw_pointer_cast(&theta_[0]),
               sizeof(float) * theta_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&gamma_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gamma_[0]),
               sizeof(float) * gamma_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&s_d_[deviceID][0]),
               thrust::raw_pointer_cast(&s_[0]), sizeof(float) * s_.size(),
               cudaMemcpyHostToDevice, streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&alphaCircle_d_[deviceID][0]),
               thrust::raw_pointer_cast(&alphaCircle_[0]),
               sizeof(float) * alphaCircle_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaAfterRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaAfterRay1_[0]),
               sizeof(int) * thetaAfterRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaAfterRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaAfterRay2_[0]),
               sizeof(int) * thetaAfterRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaBeforeRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaBeforeRay1_[0]),
               sizeof(int) * thetaBeforeRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaBeforeRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaBeforeRay2_[0]),
               sizeof(int) * thetaBeforeRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaAfterRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaAfterRay1_[0]),
               sizeof(int) * gammaAfterRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaAfterRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaAfterRay2_[0]),
               sizeof(int) * gammaAfterRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaBeforeRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaBeforeRay1_[0]),
               sizeof(int) * gammaBeforeRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaBeforeRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaBeforeRay2_[0]),
               sizeof(int) * gammaBeforeRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaGoalRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaGoalRay1_[0]),
               sizeof(float) * gammaGoalRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&gammaGoalRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&gammaGoalRay2_[0]),
               sizeof(float) * gammaGoalRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaGoalRay1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaGoalRay1_[0]),
               sizeof(float) * thetaGoalRay1_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(
               thrust::raw_pointer_cast(&thetaGoalRay2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&thetaGoalRay2_[0]),
               sizeof(float) * thetaGoalRay2_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&ray1_d_[deviceID][0]),
               thrust::raw_pointer_cast(&ray1_[0]), ray1_.size() * sizeof(int),
               cudaMemcpyHostToDevice, streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(thrust::raw_pointer_cast(&ray2_d_[deviceID][0]),
               thrust::raw_pointer_cast(&ray2_[0]), ray2_.size() * sizeof(int),
               cudaMemcpyHostToDevice, streams_[deviceID]));
}

/**
 * All values needed for setting up the class are read from the config file
 * in this function.
 *
 * @param[in] configFile path to config file
 *
 * @return returns true, if configuration file could be read successfully, else false
 */
auto Fan2Para::readConfig(const std::string& configFile) -> bool {
   int scanRate, samplingRate;
   ConfigReader configReader = ConfigReader(configFile.data());
   if (configReader.lookupValue("numberOfParallelProjections", params_.numberOfParallelProjections_)
         && configReader.lookupValue("numberOfParallelDetectors", params_.numberOfParallelDetectors_)
         && configReader.lookupValue("numberOfFanDetectors", params_.numberOfFanDetectors_)
         && configReader.lookupValue("samplingRate", samplingRate)
         && configReader.lookupValue("scanRate", scanRate)
         && configReader.lookupValue("numberOfPlanes", params_.numberOfPlanes_)
         && configReader.lookupValue("sourceOffset", params_.sourceOffset_)
         && configReader.lookupValue("detectorDiameter", params_.detectorDiameter_)
         && configReader.lookupValue("imageCenterX", params_.imageCenterX_)
         && configReader.lookupValue("imageCenterY", params_.imageCenterY_)
         && configReader.lookupValue("imageWidth", params_.imageWidth_)
         && configReader.lookupValue("blockSize1D_fan2Para",
               blockSize1D_)
         && configReader.lookupValue("blockSize2D_fan2Para", blockSize2D_)
         && configReader.lookupValue("memPoolSize_fan2Para", memPoolSize_)) {
      params_.numberOfFanProjections_ = samplingRate * 1000000 / scanRate;
      params_.rDetector_ = params_.detectorDiameter_ / 2.0;
      params_.numberOfParallelProjections_ *= 2;
      for (auto i = 0; i < params_.numberOfPlanes_; i++) {
         configReader.lookupValue("sourceDiameter", i, sourceDiam_[i]);
         configReader.lookupValue("deltaX", i, deltaX_[i]);
         configReader.lookupValue("deltaZ", i, deltaZ_[i]);
         configReader.lookupValue("sourceAngle", i, sourceAngle_[i]);
         rTarget_[i] = sourceDiam_[i] / 2.0;
      }
      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;
}

}
}
