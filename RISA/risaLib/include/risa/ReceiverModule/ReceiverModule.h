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

   enum transportProtocol: short {
      UDP,
      TCP
   };

   //! Each ReceiverModule is bound to one DetectorModule.
   /**
    * It receives the packets via an interconnection network. So far, UDP and TCP transport protocols
    * are implemented. The number of projections per packet needs to be a multiple of the number of projections per sinogram
    */
class ReceiverModule {
public:
   ReceiverModule(const std::string& address, const std::string& configPath, const int moduleID,
         std::vector<unsigned short>& buffer, OnlineReceiverNotification& notification);

   auto run() -> void;
   auto stop() -> void {run_ = false;}

private:

   UDPServer udpServer_;   //!< the class, which performs the UDP transaction

   std::vector<unsigned short>& buffer_;  //!< the input buffer, which stores a configurable amount of sinograms

   std::size_t lastIndex_{0u};   //!< identifier to check, whether a packet was lost

   int numberOfDetectorModules_; //!< the number of detector modules
   int numberOfDetectors_;        //!< the number of detectors in the fan beam sinogram
   int numberOfProjections_;     //!< the number of projections in the fan beam sinogram
   int numberOfProjectionsPerPacket_;  //!< the number of projections per packet
   int numberOfDetectorsPerModule_; //!<  the number of detectors per module

   int port_;  //!< the port, to which the object listens
   std::string address_; //!< the ip address of the sender
   transportProtocol transportProtocol_;  //!< specifies the prefered transport protocol
   int timeout_;  //!< after this duration in s, the connection is closed

   OnlineReceiverNotification& notification_;

   bool run_;

   int moduleID_;

   unsigned int bufferSize_;

   auto readConfig(const std::string& configFile) -> bool;

};

}

#endif /* RECEIVERMODULE_H_ */
