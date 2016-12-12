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
 * Authors: Tobias Frust <t.frust@hzdr.de>
 *
 */

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include "../ReceiverModule/ReceiverModule.h"
#include "OnlineReceiverNotification.h"

#include <glados/Queue.h>
#include <glados/Image.h>
#include <glados/cuda/HostMemoryManager.h>

#include <vector>
#include <thread>
#include <map>
#include <atomic>
#include <queue>

namespace risa {

//! This class controls the ReceiverModule objects.
/**
 * It gets notified, when a new sinogram arrived. If all ReceiverModules notified this class,
 * it pushes the complete sinogram through the software pipeline. (If the pipeline is reader, otherwise
 * it skips sinogramms).
 */
class Receiver {

public:
   using manager_type = glados::cuda::HostMemoryManager<unsigned short, glados::cuda::async_copy_policy>;

public:
   Receiver(const std::string& address, const std::string& configPath);

   //! #loadImage is called, when the software pipeline is able to process a new image
   /**
    * @return  the image that is pushed through the software pipeline
    */
   auto loadImage() -> glados::Image<manager_type>;

   auto run() -> void;
private:
   std::vector<ReceiverModule> modules_;

   std::map<unsigned int, std::vector<unsigned short>> buffers_;  //!< one input buffer for each detector module

   std::vector<std::thread> moduleThreads_;  //!< stores the ReceiverModule threads

   OnlineReceiverNotification notification_; //!< performs the synchronization between the Receiver and the ReceiverModules, to know when a complete sinogramm is ready

   std::array<std::atomic<int>, 27> lastEntry_; //!< stores the last indices of the received sinogramms

   int numberOfDetectorModules_; //!< the number of detector modules
   int numberOfDetectors_;       //!< the number of detectors in the fan beam sinogram
   int numberOfProjections_;     //!< the number of projections in the fan beam sinogram

   unsigned int memoryPoolIndex_;

   unsigned int bufferSize_;

   auto readConfig(const std::string& configFile) -> bool;

};

}

#endif /* RECEIVER_H_ */
