/*
 * Copyright 2016
 *
 * Receiver.cpp
 *
 *  Created on: 05.07.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include <risa/ConfigReader/ConfigReader.h>

#include <risa/Receiver/Receiver.h>

#include <ddrf/MemoryPool.h>

#include <iostream>

namespace risa {

Receiver::Receiver(const std::string& address, const std::string& configPath) :
   numberOfDetectorModules_{27},
   bufferSize_{2000},
   notification_{0}{

   if (readConfig(configPath)) {
      BOOST_LOG_TRIVIAL(error) << "Configuration file could not be read successfully. Please check!";
      throw std::runtime_error("Receiver: Configuration file could not be loaded successfully. Please check!");
   }

   for(auto i = 0u; i < numberOfDetectorModules_; i++){
      BOOST_LOG_TRIVIAL(debug) << "Creating receivermodule: " << i;
      buffers_.emplace(std::piecewise_construct, std::make_tuple(i), std::make_tuple(bufferSize_*(numberOfDetectors_/numberOfDetectorModules_)*numberOfProjections_));
   }

   modules_.reserve(numberOfDetectorModules_);
   for(auto i = 0; i < numberOfDetectorModules_; i++){
      modules_.emplace_back(address, configPath, i, buffers_[i], notification_);
   }

   ddrf::MemoryPool<manager_type>::instance()->registerStage(100, numberOfDetectors_*numberOfProjections_);

   for(auto i = 0u; i < numberOfDetectorModules_; i++){
      std::function<void(void)> f = [=]() {
         modules_[i].run();
      };
      moduleThreads_.emplace_back(f);
   }

   for(auto i = 0u; i < numberOfDetectorModules_; i++){
      moduleThreads_[i].detach();
   }
}

auto Receiver::run() -> void {

}

auto Receiver::loadImage() -> ddrf::Image<manager_type> {
   int numberOfDetectorsPerModule = 16;
   //create sinograms here
   std::size_t index = notification_.fetch();
   if(index == -1) return ddrf::Image<manager_type>();
   auto sino = ddrf::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);

   for(auto detModInd = 0; detModInd < numberOfDetectorModules_; detModInd++){
      std::size_t startIndex = (index%bufferSize_) * numberOfDetectorsPerModule*numberOfProjections_;
      std::copy(buffers_[detModInd].cbegin() + startIndex,
            buffers_[detModInd].cbegin() + startIndex + numberOfDetectorsPerModule*numberOfProjections_,
            sino.container().get() + detModInd * numberOfDetectorsPerModule * numberOfProjections_);
   }

//#pragma omp parallel for collapse(2)
//   for (auto detModInd = 0; detModInd < numberOfDetectorModules_; detModInd++) {
//      for (auto projInd = 0; projInd < numberOfProjections_; projInd++) {
//         std::size_t startIndex = projInd * numberOfDetectorsPerModule
//               + (index%bufferSize_) * numberOfDetectorsPerModule * numberOfProjections_;
//         std::size_t indexSorted = detModInd * numberOfDetectorsPerModule + projInd * numberOfDetectors_;
//         std::copy(buffers_[detModInd].begin() + startIndex, buffers_[detModInd].begin() + startIndex
//               + numberOfDetectorsPerModule, sino.container().get() + indexSorted);
//      }
//   }
   sino.setIdx(index);
   sino.setPlane(index%2);

   return std::move(sino);
}

auto Receiver::readConfig(const std::string& configFile) -> bool {
  ConfigReader configReader = ConfigReader(configFile.data());
  int samplingRate, scanRate;
  if (configReader.lookupValue("samplingRate", samplingRate)
        && configReader.lookupValue("numberOfFanDetectors", numberOfDetectors_)
        && configReader.lookupValue("scanRate", scanRate)) {
     numberOfProjections_ = samplingRate * 1000000 / scanRate;
     return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

}
