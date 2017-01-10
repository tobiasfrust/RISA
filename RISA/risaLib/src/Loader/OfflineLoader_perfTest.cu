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

#include "../../include/risa/Loader/OfflineLoader_perfTest.h"
#include "../../include/risa/Basics/performance.h"

#include <glados/MemoryPool.h>

#include <exception>
#include <fstream>
#include <chrono>

namespace risa {

OfflineLoaderPerfTest::OfflineLoaderPerfTest(const std::string& address,
      const std::string& configFile) :
      worstCaseTime_ { 0.0 }, bestCaseTime_ { std::numeric_limits<double>::max() } {
   read_json config_reader{};
   config_reader.read(configFile);
   if (readConfig(config_reader)) {
      throw std::runtime_error(
            "recoLib::OfflineLoader: Configuration file could not be loaded successfully. Please check!");
   }

   stopFrame_ = 50000u;
   index_ = 1000u;

   memoryPoolIndex_ = glados::MemoryPool<manager_type>::instance()->registerStage(
         (numberOfFrames_ + 1) * numberOfPlanes_,
         numberOfProjections_ * numberOfDetectors_);

   readInput();
}

OfflineLoaderPerfTest::~OfflineLoaderPerfTest() {
   glados::MemoryPool<manager_type>::instance()->freeMemory(memoryPoolIndex_);
   BOOST_LOG_TRIVIAL(info)<< "recoLib::OfflineLoader: WorstCaseTime: " << worstCaseTime_ << "s; BestCaseTime: " << bestCaseTime_ << "s;";
}

auto OfflineLoaderPerfTest::loadImage() -> glados::Image<manager_type> {
   if (buffer_.empty())
      return glados::Image<manager_type>();
   auto sino = std::move(buffer_.front());
   if (sino.index() > 0) {
      tmr_.stop();
      double duration = tmr_.elapsed();
      if (duration < bestCaseTime_)
         bestCaseTime_ = duration;
      if (duration > worstCaseTime_)
         worstCaseTime_ = duration;
   }
   buffer_.pop();
   if (index_ < stopFrame_) {
      auto img = glados::MemoryPool<manager_type>::instance()->requestMemory(
            memoryPoolIndex_);
      img.setIdx(index_);
      buffer_.push(std::move(img));
      index_++;
   }
   tmr_.start();
   sino.setStart(std::chrono::high_resolution_clock::now());
   return sino;
}

auto OfflineLoaderPerfTest::readInput() -> void {
   Timer tmr1, tmr2;
   std::vector<std::vector<unsigned short>> fileContents(
         numberOfDetectorModules_);
   int numberOfDetPerModule = numberOfDetectors_ / numberOfDetectorModules_;
   if (path_.back() != '/')
      path_.append("/");
   tmr1.start();
   tmr2.start();
#pragma omp parallel for default(shared) num_threads(numberOfDetectorModules_/3)
   for (auto i = 1; i <= numberOfDetectorModules_; i++) {
      std::vector<unsigned short> content;
      std::ifstream input(path_ + fileName_ + std::to_string(i) + fileEnding_,
            std::ios::in | std::ios::binary);
      if (!input) {
         BOOST_LOG_TRIVIAL(error)<< "recoLib::OfflineLoader: Source file could not be loaded.";
         throw std::runtime_error("File could not be opened. Please check!");
      }
      //allocate memory in vector
      std::streampos fileSize;
      input.seekg(0, std::ios::end);
      fileSize = input.tellg();
      input.seekg(0, std::ios::beg);
      content.resize(fileSize / sizeof(unsigned short));
      input.read((char*) &content[0], fileSize);
      fileContents[i - 1] = content;
   }
   tmr2.stop();
   for (unsigned int i = 0; i < numberOfFrames_; i++) {
      for (auto planeInd = 0; planeInd < numberOfPlanes_; planeInd++) {
         auto sino = glados::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);
         for (auto detModInd = 0; detModInd < numberOfDetectorModules_;
               detModInd++) {
            std::size_t startIndex = (planeInd + i * numberOfPlanes_) * numberOfDetPerModule * numberOfProjections_;
            std::copy(fileContents[detModInd].cbegin() + startIndex, fileContents[detModInd].cbegin() + startIndex
                        + numberOfDetPerModule * numberOfProjections_, sino.container().get()
                        + detModInd * numberOfDetPerModule * numberOfProjections_);
         }
         sino.setIdx(planeInd + i * numberOfPlanes_);
         sino.setPlane(planeInd);
         buffer_.push(std::move(sino));
      }
   }
   tmr1.stop();
   double totalFileSize = numberOfProjections_ * numberOfDetectors_
         * numberOfPlanes_ * numberOfFrames_ * sizeof(unsigned short) / 1024.0
         / 1024.0;
   BOOST_LOG_TRIVIAL(info)<< "recoLib::OfllineLoader: Reading and sorting data input took " << tmr1.elapsed() << " s, " << totalFileSize/tmr2.elapsed() << " MByte/s.";
}

/**
 * All values needed for setting up the class are read from the config file
 * in this function.
 *
 * @param[in] configFile path to config file
 *
 * @return returns true, if configuration file could be read successfully, else false
 */
auto OfflineLoaderPerfTest::readConfig(const read_json& config_reader) -> bool {
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

