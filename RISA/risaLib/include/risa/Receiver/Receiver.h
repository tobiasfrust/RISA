/*
 * Copyright 2016
 *
 * Receiver.h
 *
 *  Created on: 05.07.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include "../ReceiverModule/ReceiverModule.h"
#include "OnlineReceiverNotification.h"

#include <ddrf/Queue.h>
#include <ddrf/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>

#include <vector>
#include <thread>
#include <map>
#include <atomic>
#include <queue>

namespace risa {

class Receiver {

public:
   using manager_type = ddrf::cuda::HostMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

public:
   Receiver(const std::string& address, const std::string& configPath);


   auto loadImage() -> ddrf::Image<manager_type>;

   auto run() -> void;
private:
   std::vector<ReceiverModule> modules_;

   std::map<unsigned int, std::vector<unsigned short>> buffers_;

   std::vector<std::thread> moduleThreads_;

   OnlineReceiverNotification notification_;

   std::array<std::atomic<int>, 27> lastEntry_;

   int numberOfDetectorModules_;
   int numberOfDetectors_;
   int numberOfProjections_;

   unsigned int memoryPoolIndex_;

   unsigned int bufferSize_;

   auto readConfig(const std::string& configFile) -> bool;

};

}

#endif /* RECEIVER_H_ */
