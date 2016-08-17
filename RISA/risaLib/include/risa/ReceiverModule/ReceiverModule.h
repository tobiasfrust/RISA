/*
 * Copyright 2016
 *
 * ReceiverModule.h
 *
 *  Created on: 05.07.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef RECEIVERMODULE_H_
#define RECEIVERMODULE_H_

#include "../UDPServer/UDPServer.h"
#include "../Receiver/OnlineReceiverNotification.h"

#include <ddrf/Queue.h>

#include <vector>
#include <thread>
#include <atomic>

namespace risa {

class ReceiverModule {
public:
   ReceiverModule(const std::string& address, const std::string& configPath, const int moduleID,
         std::vector<unsigned short>& buffer, OnlineReceiverNotification& notification);

   auto run() -> void;
   auto stop() -> void {run_ = false;}

private:

   std::vector<unsigned short>& buffer_;

   std::size_t lastIndex_;

   int numberOfDetectorModules_;
   int numberOfDetectors_;
   int numberOfProjections_;

   int port_;

   OnlineReceiverNotification& notification_;

   bool run_;

   int moduleID_;

   unsigned int bufferSize_;

   auto readConfig(const std::string& configFile) -> bool;

};

}

#endif /* RECEIVERMODULE_H_ */
