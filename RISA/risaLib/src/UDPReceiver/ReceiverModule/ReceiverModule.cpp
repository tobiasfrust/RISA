/*
 * Copyright 2016
 *
 * Receiver.cpp
 *
 *  Created on: 05.07.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include <risa/ReceiverModule/ReceiverModule.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <boost/log/trivial.hpp>
#include <boost/asio.hpp>

namespace risa {

using boost::asio::ip::tcp;

ReceiverModule::ReceiverModule(const std::string& address, const std::string& configPath, const int moduleID,
      std::vector<unsigned short>& buffer, OnlineReceiverNotification& notification) :
   bufferSize_{1999},
   numberOfDetectorModules_{27},
   moduleID_{moduleID},
   buffer_(buffer),
   run_{true},
   notification_(notification),
   port_{4000+moduleID}
   {

   if (readConfig(configPath)) {
      BOOST_LOG_TRIVIAL(error) << "Configuration file could not be read successfully. Please check!";
      throw std::runtime_error("ReceiverModule: Configuration file could not be loaded successfully. Please check!");
   }

   BOOST_LOG_TRIVIAL(debug) << "Created Module receiving at " << address << " listening to port " << 4000+moduleID;
}

auto ReceiverModule::run() -> void {
   BOOST_LOG_TRIVIAL(debug) << "Test";

   std::vector<unsigned short> buf(numberOfProjections_*16 + sizeof(std::size_t)/sizeof(unsigned short));
   unsigned int sinoSize = numberOfProjections_*16;
   std::size_t headerSize{sizeof(std::size_t)/sizeof(unsigned short)};
   int numBytes = 0;
   int timeoutS = 30;

   boost::asio::io_service io_service;

   tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port_));

   tcp::socket socket(io_service);
   acceptor.accept(socket);

   boost::system::error_code error;

   while(true){
      boost::asio::read(socket, boost::asio::buffer(buf.data(), buf.size()*sizeof(unsigned short)), error);
      if (error == boost::asio::error::eof)
         break; // Connection closed cleanly by peer.
      std::size_t index = *((std::size_t *)buf.data());
      BOOST_LOG_TRIVIAL(debug) << "ReceiverModule " << moduleID_ << " received packet " << index;
      std::copy(buf.cbegin() + headerSize, buf.cend(), buffer_.begin() + sinoSize * (index%bufferSize_));
      notification_.notify(moduleID_, index);
   }
   notification_.notify(moduleID_, -1);
   BOOST_LOG_TRIVIAL(info) << "ReceiverModul " << moduleID_ << ": No packets arriving since " << timeoutS << "s. Finishing.";
}

auto ReceiverModule::readConfig(const std::string& configFile) -> bool {
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
