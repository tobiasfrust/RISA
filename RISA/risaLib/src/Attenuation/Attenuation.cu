/*
 * Copyright 2016
 *
 * Attenuation.cu
 *
 *  Created on: 02.06.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include "../DetectorInterpolation/interpolationFunctions.h"

#include <risa/Attenuation/Attenuation.h>
#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Basics/performance.h>

#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <omp.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iterator>
#include <exception>
#include <pthread.h>

namespace risa {
namespace cuda {

Attenuation::Attenuation(const std::string& configFile) {

   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::cuda::Attenuation: Configuration file could not be loaded successfully. Please check!");
   }

   numberOfDarkFrames_ = 500;

   CHECK(cudaGetDeviceCount(&numberOfDevices_));

   //custom streams are necessary, because profiling with nvprof not possible with
   //-default-stream per-thread option
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      memoryPoolIdxs_[i] =
            ddrf::MemoryPool<deviceManagerType>::instance()->registerStage(memPoolSize_,
                  numberOfDetectors_ * numberOfProjections_);
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      streams_[i] = stream;
   }

   init();

   //initialize worker threads
   for (auto i = 0; i < numberOfDevices_; i++) {
      processorThreads_[i] = std::thread { &Attenuation::processor, this, i };
   }

   BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Attenuation: Running " << numberOfDevices_ << " Threads.";
}

Attenuation::~Attenuation() {
   for (auto idx : memoryPoolIdxs_) {
      CHECK(cudaSetDevice(idx.first));
      ddrf::MemoryPool<deviceManagerType>::instance()->freeMemory(idx.second);
   }
   for (auto i = 0; i < numberOfDevices_; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(streams_[i]));
   }
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Attenuation: Destroyed.";
}

auto Attenuation::process(input_type&& sinogram) -> void {
   if (sinogram.valid()) {
      BOOST_LOG_TRIVIAL(debug)<< "Attenuation: Image arrived with Index: " << sinogram.index() << "to device " << sinogram.device();
      sinograms_[sinogram.device()].push(std::move(sinogram));
   } else {
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Attenuation: Received sentinel, finishing.";

      //send sentinal to processor thread and wait 'til it's finished
      for(auto i = 0; i < numberOfDevices_; i++) {
         sinograms_[i].push(input_type());
      }

      for(auto i = 0; i < numberOfDevices_; i++) {
         processorThreads_[i].join();
      }
      //push sentinel to results for next stage
      results_.push(output_type());
      BOOST_LOG_TRIVIAL(info) << "recoLib::cuda::Attenuation: Finished.";
   }
}

auto Attenuation::wait() -> output_type {
   return results_.take();
}

auto Attenuation::processor(const int deviceID) -> void {
   //nvtxNameOsThreadA(pthread_self(), "Attenuation");
   CHECK(cudaSetDevice(deviceID));
   auto avgDark_d = ddrf::cuda::make_device_ptr<float>(avgDark_.size());
   auto avgReference_d = ddrf::cuda::make_device_ptr<float>(
         avgReference_.size());
   auto mask_d = ddrf::cuda::make_device_ptr<float>(
         numberOfDetectors_ * numberOfProjections_);
   CHECK(
         cudaMemcpyAsync(avgDark_d.get(), avgDark_.data(),
               sizeof(float) * avgDark_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   CHECK(
         cudaMemcpyAsync(avgReference_d.get(), avgReference_.data(),
               sizeof(float) * avgReference_.size(), cudaMemcpyHostToDevice,
               streams_[deviceID]));
   //compute mask for relevant area
   std::vector<float> mask;
   relevantAreaMask(mask);
   CHECK(
         cudaMemcpyAsync(mask_d.get(), mask.data(), sizeof(float) * mask.size(),
               cudaMemcpyHostToDevice, streams_[deviceID]));

   dim3 blocks(blockSize2D_, blockSize2D_);
   dim3 grids(std::ceil(numberOfDetectors_ / (float)blockSize2D_),
         std::ceil(numberOfProjections_ / (float)blockSize2D_));
   float temp = pow(10, -5);
   CHECK(cudaStreamSynchronize(streams_[deviceID]));
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Attenuation: Running Thread for Device " << deviceID;

   while (true) {
      auto sinogram = sinograms_[deviceID].take();
      if (!sinogram.valid())
         break;
      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Attenuation: Attenuationing image with Index " << sinogram.index();

      auto sino =
            ddrf::MemoryPool<deviceManagerType>::instance()->requestMemory(
                  memoryPoolIdxs_[deviceID]);

      computeAttenuation<<<grids, blocks, 0, streams_[deviceID]>>>(
            sinogram.container().get(), mask_d.get(), sino.container().get(),
            avgReference_d.get(), avgDark_d.get(), temp, numberOfDetectors_,
            numberOfProjections_, sinogram.plane());
      CHECK(cudaPeekAtLastError());

      sino.setIdx(sinogram.index());
      sino.setDevice(deviceID);
      sino.setPlane(sinogram.plane());
      sino.setStart(sinogram.start());

      //wait until work on device is finished
      CHECK(cudaStreamSynchronize(streams_[deviceID]));
      results_.push(std::move(sino));

      BOOST_LOG_TRIVIAL(debug)<< "recoLib::cuda::Attenuation: Attenuationing image with Index " << sinogram.index() << " finished.";
   }
}

auto Attenuation::init() -> void {
   //create filter function
   std::vector<double> filterFunction{0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 3.0, 3.5, 2.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5};
   double sum = std::accumulate(filterFunction.cbegin(), filterFunction.cend(), 0.0);
   std::transform(filterFunction.begin(), filterFunction.end(), filterFunction.begin(),
         std::bind1st(std::multiplies<double>(), 1.0/sum));

   //read and average reference input values
   std::vector<unsigned short> referenceValues;
   if(pathReference_.back() != '/')
      pathReference_.append("/");
   std::string refPath = pathReference_ + "ref_empty_tomograph_repaired_DetModNr_";
   readInput(refPath, referenceValues, numberOfRefFrames_);
   //interpolate reference measurement
   for(auto i = 0; i < numberOfRefFrames_*numberOfPlanes_; i++){
      std::vector<int> defectDetectors(numberOfProjections_*numberOfDetectors_);
      findDefectDetectors(referenceValues.data()+i*numberOfDetectors_*numberOfProjections_, filterFunction, defectDetectors, numberOfDetectors_, numberOfProjections_,
         threshMin_, threshMax_);
      interpolateDefectDetectors(referenceValues.data()+i*numberOfDetectors_*numberOfProjections_, defectDetectors, numberOfDetectors_, numberOfProjections_);
   }
   computeAverage(referenceValues, avgReference_);

   //read and average dark input values
   std::vector<unsigned short> darkValues;
   if(pathDark_.back() != '/')
      pathDark_.append("/");
   std::string darkPath = pathDark_ + "dark_192.168.100_DetModNr_";
   readInput(darkPath, darkValues, numberOfDarkFrames_);
   computeDarkAverage(darkValues, avgDark_);
   //interpolate dark average
   for(auto j = 0; j < numberOfPlanes_; j++){
      for(auto i = 0; i < numberOfDetectors_; i++){
         if(avgDark_[i + j * numberOfDetectors_] > 300.0){
            BOOST_LOG_TRIVIAL(info) << "Interpolating dark value at detector " << i << " in plane " << j;
            avgDark_[numberOfDetectors_ * j + i] =
                                 0.5 * (avgDark_[numberOfDetectors_ * j + (i + 1)%numberOfDetectors_] +
                                       avgDark_[numberOfDetectors_ * j + (i - 1)%numberOfDetectors_]);
         }
      }
   }
}

template <typename T>
auto Attenuation::computeDarkAverage(const std::vector<T>& values, std::vector<float>& average) -> void {
   average.resize(numberOfDetectors_*numberOfPlanes_, 0.0);
   float factor = 1.0/ (float)((float)numberOfDarkFrames_*(float)numberOfProjections_);
   factor = 0.0;
   for(auto i = 0; i < numberOfDarkFrames_; i++){
      for(auto planeInd = 0; planeInd < numberOfPlanes_; planeInd++){
         for(auto detInd = 0; detInd < numberOfDetectors_; detInd++){
            for(auto projInd = 0; projInd < numberOfProjections_; projInd++){
               const float val = (float)values[detInd + numberOfDetectors_*projInd + (i*numberOfPlanes_+planeInd)*numberOfDetectors_*numberOfProjections_];
               average[detInd + planeInd*numberOfDetectors_] += val * factor;
            }
         }
      }
   }
}

template<typename T>
auto Attenuation::computeAverage(const std::vector<T>& values,
      std::vector<float>&average) -> void {
   average.resize(numberOfProjections_ * numberOfDetectors_ * numberOfPlanes_);
   float factor = 1.0 / (float) numberOfRefFrames_;
   for (auto i = 0; i < numberOfRefFrames_; i++) {
      for (auto planeInd = 0; planeInd < numberOfPlanes_; planeInd++) {
         for (auto index = 0; index < numberOfDetectors_ * numberOfProjections_;
               index++) {
            average[index + planeInd * numberOfDetectors_ * numberOfProjections_] +=
                  values[(i + planeInd) * numberOfProjections_
                        * numberOfDetectors_ + index] * factor;
         }
      }
   }
}

template<typename T>
auto Attenuation::readDarkInputFiles(std::string& path,
      std::vector<T>& values) -> void {
   //if(path.back() != '/')
   //   path.append("/");
   std::ifstream input(path + "dark_192.168.100.fxc",
         std::ios::in | std::ios::binary);
   if (!input) {
      BOOST_LOG_TRIVIAL(error)<< "recoLib::cuda::Attenuation: Source file could not be loaded.";
      throw std::runtime_error("File could not be opened. Please check!");
   }
   //allocate memory in vector
   std::streampos fileSize;
   input.seekg(0, std::ios::end);
   fileSize = input.tellg();
   input.seekg(0, std::ios::beg);
   values.resize(numberOfDetectors_ * numberOfPlanes_);
   input.read((char*) &values[0],
         numberOfDetectors_ * numberOfPlanes_ * sizeof(T));
}

template<typename T>
auto Attenuation::readInput(std::string& path,
      std::vector<T>& values, const int numberOfFrames) -> void {
   std::vector<std::vector<T>> fileContents(numberOfDetectorModules_);
   Timer tmr1, tmr2;
   //if(path.back() != '/')
   //   path.append("/");
   tmr1.start();
   tmr2.start();
#pragma omp parallel for default(shared) num_threads(9)
   for (auto i = 1; i <= numberOfDetectorModules_; i++) {
      std::vector<T> content;
      //TODO: make filename and ending configurable
      std::ifstream input(path + std::to_string(i) + ".fx", std::ios::in | std::ios::binary);
      if (!input) {
         BOOST_LOG_TRIVIAL(error)<< "recoLib::cuda::Attenuation: Source file " << path + std::to_string(i) + ".fx" << " could not be loaded.";
         throw std::runtime_error("File could not be opened. Please check!");
      }
      //allocate memory in vector
      std::streampos fileSize;
      input.seekg(0, std::ios::end);
      fileSize = input.tellg();
      input.seekg(0, std::ios::beg);
      content.resize(fileSize / sizeof(T));
      input.read((char*) &content[0], fileSize);
      fileContents[i - 1] = content;
   }
   tmr2.stop();
   int numberOfDetPerModule = numberOfDetectors_ / numberOfDetectorModules_;
   values.resize(fileContents[0].size() * numberOfDetectorModules_);
   for (auto i = 0; i < numberOfFrames; i++) {
      for (auto planeInd = 0; planeInd < numberOfPlanes_; planeInd++) {
         for (auto projInd = 0; projInd < numberOfProjections_; projInd++) {
            for (auto detModInd = 0; detModInd < numberOfDetectorModules_;
                  detModInd++) {
               unsigned int startIndex = projInd * numberOfDetPerModule
                     + (planeInd + i * numberOfPlanes_) * numberOfDetPerModule * numberOfProjections_;
               unsigned int indexSorted = detModInd * numberOfDetPerModule
                     + projInd * numberOfDetectors_
                     + (planeInd + i * numberOfPlanes_) * numberOfDetectors_ * numberOfProjections_;
               std::copy(fileContents[detModInd].begin() + startIndex,
                     fileContents[detModInd].begin() + startIndex
                           + numberOfDetPerModule,
                     values.begin() + indexSorted);
            }
         }
      }
   }
   tmr1.stop();
   double totalFileSize = numberOfProjections_*numberOfDetectors_*numberOfPlanes_*numberOfRefFrames_*sizeof(unsigned short)/1024.0/1024.0;
   BOOST_LOG_TRIVIAL(info)<< "recoLib::cuda::Attenuation: Reading and sorting reference input took " << tmr1.elapsed() << " s, " << totalFileSize/tmr2.elapsed() << " MByte/s.";
}

template<typename T>
auto Attenuation::relevantAreaMask(std::vector<T>& mask) -> void {
   unsigned int ya, yb, yc, yd, ye;
   unsigned int yMin, yMax;
   double lowerLimit = (lowerLimOffset_ + sourceOffset_) / 360.0;
   double upperLimit = (upperLimOffset_ + sourceOffset_) / 360.0;
   //fill whole mask with ones and mask out the unrelevant parts afterwards
   mask.resize(numberOfProjections_ * numberOfDetectors_);
   std::fill(mask.begin(), mask.end(), 1.0);

   ya = std::round(lowerLimit * numberOfProjections_);
   yb = ya;
   yc = std::round(upperLimit * numberOfProjections_);
   yd = yc;

   //slope of the straight
   double m = ((double)ya - (double)yd) / ((double)xa_ - (double)xd_);

   ye = std::round((double)yc + ((double)xe_ - (double)xc_) * m);

   for (unsigned int x = 0; x <= xa_; x++) {
      yMin = ya;
      yMax = std::round(ye + m * x);
      for (auto y = yMin; y < yMax; y++)
         mask[x + y * numberOfDetectors_] = 0.0;
   }

   for (auto x = xa_; x <= xc_; x++) {
      yMin = std::round(ya + m * (x - xa_));
      yMax = std::round(ye + m * x);
      for (auto y = yMin; y < yMax; y++)
         mask[x + y * numberOfDetectors_] = 0.0;
   }

   for (auto x = xc_; x <= xd_; x++) {
      yMin = std::round(ya + m * (x - xa_));
      yMax = yd;
      for (auto y = yMin; y < yMax; y++)
         mask[x + y * numberOfDetectors_] = 0.0;
   }

   for (auto x = xb_; x <= xf_; x++) {
      yMin = yb;
      yMax = std::round(yb + m * (x - xb_));
      for (auto y = yMin; y < yMax; y++)
         mask[x + y * numberOfDetectors_] = 0.0;
   }

   std::fill(mask.begin(),
         mask.begin() + lowerLimit * numberOfDetectors_ * numberOfProjections_,
         0.0);
   std::fill(
         mask.begin() + upperLimit * numberOfProjections_ * numberOfDetectors_,
         mask.end(), 0.0);
}

auto Attenuation::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(
         configFile.data());
   int samplingRate, scanRate;
   if (configReader.lookupValue("numberOfFanDetectors", numberOfDetectors_)
         && configReader.lookupValue("numberOfDetectorModules", numberOfDetectorModules_)
         && configReader.lookupValue("numberOfReferenceFrames", numberOfRefFrames_)
         && configReader.lookupValue("darkInputPath", pathDark_)
         && configReader.lookupValue("referenceInputPath", pathReference_)
         && configReader.lookupValue("numberOfPlanes", numberOfPlanes_)
         && configReader.lookupValue("samplingRate", samplingRate)
         && configReader.lookupValue("scanRate", scanRate)
         && configReader.lookupValue("sourceOffset", sourceOffset_)
         && configReader.lookupValue("xa", xa_)
         && configReader.lookupValue("xb", xb_)
         && configReader.lookupValue("xc", xc_)
         && configReader.lookupValue("xd", xd_)
         && configReader.lookupValue("xe", xe_)
         && configReader.lookupValue("xf", xf_)
         && configReader.lookupValue("lowerLimOffset", lowerLimOffset_)
         && configReader.lookupValue("upperLimOffset", upperLimOffset_)
         && configReader.lookupValue("blockSize2D_attenuation", blockSize2D_)
         && configReader.lookupValue("memPoolSize_attenuation", memPoolSize_)
         && configReader.lookupValue("thresh_min", threshMin_)
         && configReader.lookupValue("thresh_max", threshMax_)) {
      numberOfProjections_ = samplingRate * 1000000 / scanRate;
      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;
}

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
      const int numberOfProjections, const int planeId) {

   auto x = ddrf::cuda::getX();
   auto y = ddrf::cuda::getY();
   if (x >= numberOfDetectors || y >= numberOfProjections)
      return;

   auto sinoIndex = numberOfDetectors * y + x;

   float numerator = (float) (sinogram_in[sinoIndex])
         - avgDark[planeId * numberOfDetectors + x];

   float denominator = avgReference[planeId * numberOfDetectors * numberOfProjections + sinoIndex]
         - avgDark[planeId * numberOfDetectors + x];

   if (numerator < temp)
      numerator = temp;
   if (denominator < temp)
      denominator = temp;

   //comutes the attenuation and multiplies with mask for hiding the unrelevant region
   sinogram_out[sinoIndex] = -log(numerator / denominator) * mask[sinoIndex];

}

}
}
