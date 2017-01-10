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

#include "../../../include/risa/Receiver/Receiver.h"

#include <glados/MemoryPool.h>

#include <iostream>

namespace risa {

Receiver::Receiver(const std::string& address, const std::string& configPath) : notification_{27}{

   read_json config_reader{};
   config_reader.read(configPath);
   if (readConfig(config_reader)) {
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

   memoryPoolIndex_ = glados::MemoryPool<manager_type>::instance()->registerStage(100, numberOfDetectors_*numberOfProjections_);

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

auto Receiver::loadImage() -> glados::Image<manager_type> {
   int numberOfDetectorsPerModule = 16;
   //create sinograms here
   std::size_t index = notification_.fetch();
   if(index == -1) return glados::Image<manager_type>();
   auto sino = glados::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);

   for(auto detModInd = 0; detModInd < numberOfDetectorModules_; detModInd++){
      std::size_t startIndex = (index%bufferSize_) * numberOfDetectorsPerModule*numberOfProjections_;
      std::copy(buffers_[detModInd].cbegin() + startIndex,
            buffers_[detModInd].cbegin() + startIndex + numberOfDetectorsPerModule*numberOfProjections_,
            sino.container().get() + detModInd * numberOfDetectorsPerModule * numberOfProjections_);
   }
   sino.setIdx(index);
   sino.setPlane(index%2);
   sino.setStart(std::chrono::high_resolution_clock::now());

   return std::move(sino);
}

auto Receiver::readConfig(const read_json& config_reader) -> bool {
  int sampling_rate, scan_rate;
  try {
	  sampling_rate = config_reader.get_value<int>("sampling_rate");
	  numberOfDetectors_ = config_reader.get_value<int>("number_of_fan_det");
	  scan_rate = config_reader.get_value<int>("scan_rate");
	  bufferSize_ = config_reader.get_value<int>("input_buffersize");
	  numberOfDetectorModules_ = config_reader.get_value<int>("number_of_det_modules");
  } catch (const boost::property_tree::ptree_error& e) {
	  BOOST_LOG_TRIVIAL(error) << "risa::Receiver: Failed to read config: " << e.what();
	  return EXIT_FAILURE;
  }
  numberOfProjections_ = sampling_rate * 1000000 / scan_rate;
  return EXIT_SUCCESS;
}

}
