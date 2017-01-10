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

#include <risa/Loader/OfflineLoader.h>
#include <risa/Basics/performance.h>

#include <glados/MemoryPool.h>

#include <exception>
#include <chrono>

namespace risa {

OfflineLoader::OfflineLoader(const std::string& address,
		const std::string& config_file) :
      worstCaseTime_ { 0.0 }, bestCaseTime_ { std::numeric_limits<double>::max() } {

   risa::read_json config_reader{};
   config_reader.read(config_file);
   if (readConfig(config_reader)) {
      throw std::runtime_error(
            "recoLib::OfflineLoader: Configuration file could not be loaded successfully. Please check!");
   }

   numberOfDetectorsPerModule_ = numberOfDetectors_ / numberOfDetectorModules_;

   memoryPoolIndex_ = glados::MemoryPool<manager_type>::instance()->registerStage(
         250, numberOfProjections_ * numberOfDetectors_);

   readInput();
}

OfflineLoader::~OfflineLoader() {
   glados::MemoryPool<manager_type>::instance()->freeMemory(memoryPoolIndex_);
   BOOST_LOG_TRIVIAL(info)<< "recoLib::OfflineLoader: WorstCaseTime: " << worstCaseTime_ << "s; BestCaseTime: " << bestCaseTime_ << "s;";
}

auto OfflineLoader::loadImage() -> glados::Image<manager_type> {
   if (index_ >= numberOfFrames_ * numberOfPlanes_)
      return glados::Image<manager_type>();
   BOOST_LOG_TRIVIAL(debug) << "risa::Offlineloader: Loading image with index " << index_;
   auto sino = glados::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);
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
auto OfflineLoader::readConfig(const read_json& config_reader) -> bool {
   int sampling_rate, scan_rate;
   try {
	   numberOfDetectors_ = config_reader.get_value<int>("number_of_fan_detectors");
	   numberOfDetectorModules_ = config_reader.get_value<int>("number_of_det_modules");
	   path_ = config_reader.get_element_in_list<std::string, std::string>("inputs", "inputpath", std::make_pair("inputtype", "data"));
	   fileName_ = config_reader.get_element_in_list<std::string, std::string>("inputs", "file_prefix", std::make_pair("inputtype", "data"));
	   fileEnding_ = config_reader.get_element_in_list<std::string, std::string>("inputs", "file_ending", std::make_pair("inputtype", "data"));
	   numberOfPlanes_ = config_reader.get_value<int>("number_of_planes");
	   sampling_rate = config_reader.get_value<int>("sampling_rate");
	   scan_rate = config_reader.get_value<int>("scan_rate");
	   numberOfFrames_ = config_reader.get_value<int>("number_of_data_frames");
   } catch (const boost::property_tree::ptree_error& e) {
	   BOOST_LOG_TRIVIAL(error) << "risa::cuda:OfflineLoader: Failed to read config: " << e.what();
	   return EXIT_FAILURE;
   }
   numberOfProjections_ = sampling_rate * 1000000 / scan_rate;
   return EXIT_SUCCESS;
}

}

