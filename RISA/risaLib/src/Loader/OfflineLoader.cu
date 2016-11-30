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

#include <risa/ConfigReader/ConfigReader.h>
#include <risa/Loader/OfflineLoader.h>
#include <risa/Basics/performance.h>

#include <ddrf/MemoryPool.h>

#include <exception>
#include <chrono>

namespace risa {

OfflineLoader::OfflineLoader(const std::string& address,
      const std::string& configFile) :
      worstCaseTime_ { 0.0 }, bestCaseTime_ { std::numeric_limits<double>::max() } {
   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::OfflineLoader: Configuration file could not be loaded successfully. Please check!");
   }

   numberOfDetectorsPerModule_ = numberOfDetectors_ / numberOfDetectorModules_;

   memoryPoolIndex_ = ddrf::MemoryPool<manager_type>::instance()->registerStage(
         250, numberOfProjections_ * numberOfDetectors_);

   readInput();
}

OfflineLoader::~OfflineLoader() {
   ddrf::MemoryPool<manager_type>::instance()->freeMemory(memoryPoolIndex_);
   BOOST_LOG_TRIVIAL(info)<< "recoLib::OfflineLoader: WorstCaseTime: " << worstCaseTime_ << "s; BestCaseTime: " << bestCaseTime_ << "s;";
}

auto OfflineLoader::loadImage() -> ddrf::Image<manager_type> {
   if (index_ >= numberOfFrames_ * numberOfPlanes_)
      return ddrf::Image<manager_type>();
   BOOST_LOG_TRIVIAL(debug) << "risa::Offlineloader: Loading image with index " << index_;
   auto sino = ddrf::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);
#pragma omp parallel for
   for (auto detModInd = 0; detModInd < numberOfDetectorModules_; detModInd++) {
      ifstreams_[detModInd].get()->seekg(index_ * numberOfProjections_ * numberOfDetectorsPerModule_ * sizeof(unsigned short),
            std::ios::beg);
      ifstreams_[detModInd].get()->read((char*) sino.container().get()+ detModInd * numberOfDetectorsPerModule_
                        * sizeof(unsigned short) * numberOfProjections_, numberOfProjections_ * numberOfDetectorsPerModule_ * sizeof(unsigned short));
   }
//   ifstreams_[0].get()->seekg(index_ * numberOfProjections_ * numberOfDetectors_ * sizeof(float),
//         std::ios::beg);
//   ifstreams_[0].get()->read(
//         (char*) sino.container().get(), numberOfProjections_ * numberOfDetectors_ * sizeof(float));
   BOOST_LOG_TRIVIAL(debug)<< "Loading sinogram " << index_;
   sino.setIdx(index_);
   sino.setPlane(index_%numberOfPlanes_);
   index_++;
   sino.setStart(std::chrono::high_resolution_clock::now());
   return sino;
}

auto OfflineLoader::readInput() -> void {
   if (path_.back() != '/')
      path_.append("/");

   for (auto i = 1; i <= numberOfDetectorModules_; i++) {
      auto ifStream = std::unique_ptr <std::ifstream> (new std::ifstream(
                  path_ + fileName_ + std::to_string(i) + fileEnding_,
                  std::ios::in | std::ios::binary));
      ifstreams_.push_back(std::move(ifStream));
   }

//   auto ifStream = std::unique_ptr <std::ifstream> (new std::ifstream(
//               path_ + fileName_ + fileEnding_,
//               std::ios::in | std::ios::binary));
//   ifstreams_.push_back(std::move(ifStream));
}

/**
 * All values needed for setting up the class are read from the config file
 * in this function.
 *
 * @param[in] configFile path to config file
 *
 * @return returns true, if configuration file could be read successfully, else false
 */
auto OfflineLoader::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());
   int samplingRate, scanRate;
   if (configReader.lookupValue("numberOfFanDetectors", numberOfDetectors_)
         && configReader.lookupValue("numberOfDetectorModules",
               numberOfDetectorModules_)
         && configReader.lookupValue("dataInputPath", path_)
         && configReader.lookupValue("dataFileName", fileName_)
         && configReader.lookupValue("dataFileEnding", fileEnding_)
         && configReader.lookupValue("numberOfPlanes", numberOfPlanes_)
         && configReader.lookupValue("samplingRate", samplingRate)
         && configReader.lookupValue("scanRate", scanRate)
         && configReader.lookupValue("numberOfDataFrames", numberOfFrames_)) {
      numberOfProjections_ = samplingRate * 1000000 / scanRate;
      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;
}

}

