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

#include <risa/ReceiverModule/ReceiverModule.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <boost/log/trivial.hpp>
#include <boost/asio.hpp>
#include <boost/asio/buffer.hpp>

#include <future>

namespace risa {

using tcp = boost::asio::ip::tcp;

ReceiverModule::ReceiverModule(const std::string& address, const std::string& configPath, const int moduleID,
      std::vector<unsigned short>& buffer, OnlineReceiverNotification& notification) :
   moduleID_{moduleID},
   buffer_(buffer),
   run_{true},
   address_{address},
   notification_(notification),
   udpServer_{address, 4000+moduleID},
   port_{4000+moduleID}
   {

   if (readConfig(configPath)) {
      BOOST_LOG_TRIVIAL(error) << "Configuration file could not be read successfully. Please check!";
      throw std::runtime_error("ReceiverModule: Configuration file could not be loaded successfully. Please check!");
   }

   BOOST_LOG_TRIVIAL(debug) << "Created Module receiving at " << address << " listening to port " << 4000+moduleID;
}

auto ReceiverModule::run() -> void {
   std::size_t headerSize{(sizeof(std::size_t)+sizeof(unsigned short))/sizeof(unsigned short)};
   std::vector<unsigned short> buf(numberOfProjectionsPerPacket_*numberOfDetectorsPerModule_ + headerSize);
   unsigned int sinoSize = numberOfProjections_*numberOfDetectorsPerModule_;
   int numBytes = 0;
   unsigned short numberOfParts = numberOfProjections_/numberOfProjectionsPerPacket_ - 1;
   if(transportProtocol_ == transportProtocol::TCP){
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
         unsigned short partID = *((unsigned short*)(buf.data() + sizeof(std::size_t)/sizeof(unsigned short)));
         BOOST_LOG_TRIVIAL(debug) << "ReceiverModule " << moduleID_ << " received packet " << index << " part ID: " << partID;
         std::copy(buf.cbegin() + headerSize, buf.cend(), buffer_.begin() + sinoSize * (index%bufferSize_) + partID * numberOfProjectionsPerPacket_*numberOfDetectorsPerModule_);
         if(numberOfParts == partID)
            notification_.notify(moduleID_, index);
      }
   }else if(transportProtocol_ == transportProtocol::UDP){
      //Possibility how to realize timeout with boost::asio::udp was not found yet
      //perhaps a final packet needs to be send, that specifies the end of transaction
      /*boost::asio::io_service io_service;
      udp::socket socket(io_service);
      udp::endpoint listen_endpoint(boost::asio::ip::address::from_string(address_), port_);
      socket.open(listen_endpoint.protocol());
      socket.bind(listen_endpoint);*/
      while(true){
         numBytes = udpServer_.timed_recv((char*)(buf.data()), buf.size()*sizeof(unsigned short), timeout_);
         if(numBytes < 0) break;
         BOOST_LOG_TRIVIAL(debug) << "Number of bytes received: " << numBytes;
         /*boost::system::error_code ec;
         std::size_t n = socket.receive_from(boost::asio::buffer(buf.data(), buf.size()*sizeof(unsigned short)), listen_endpoint);
         */
         std::size_t index = *((std::size_t *)buf.data());
         unsigned short partID = *((unsigned short*)(buf.data() + sizeof(std::size_t)/sizeof(unsigned short)));
         int diff = index*numberOfParts + partID - lastIndex_;
         if(diff > 1){
            BOOST_LOG_TRIVIAL(warning) << "ReceiverModule " << moduleID_ << ": Lost package or wrong order. Last " << lastIndex_ << " new: " << index*numberOfParts + partID;
            lastIndex_ = index*numberOfParts + partID;
            //continue;
         }
         lastIndex_ = index*numberOfParts + partID;
         BOOST_LOG_TRIVIAL(debug) << "ReceiverModule " << moduleID_ << " received packet " << index << " partID: " << partID;
         auto it = std::count(buf.cbegin()+headerSize, buf.cend(), 0);
         std::copy(buf.cbegin() + headerSize, buf.cend(), buffer_.begin() + sinoSize * (index%bufferSize_) + partID * numberOfProjectionsPerPacket_*numberOfDetectorsPerModule_);
         if(numberOfParts == partID)
            notification_.notify(moduleID_, index);
      }
   }
   notification_.notify(moduleID_, -1);
   BOOST_LOG_TRIVIAL(info) << "ReceiverModul " << moduleID_ << ": No packets arriving since " << timeout_ << "s. Finishing.";
}

auto ReceiverModule::readConfig(const std::string& configFile) -> bool {
  ConfigReader configReader = ConfigReader(configFile.data());
  int samplingRate, scanRate;
  std::string transportProt;
  if (configReader.lookupValue("samplingRate", samplingRate)
        && configReader.lookupValue("numberOfFanDetectors", numberOfDetectors_)
        && configReader.lookupValue("scanRate", scanRate)
        && configReader.lookupValue("transportProtocol", transportProt)
        && configReader.lookupValue("timeout", timeout_)
        && configReader.lookupValue("numberOfDetectorsPerModule", numberOfDetectorsPerModule_)
        && configReader.lookupValue("numberOfProjectionsPerPacket", numberOfProjectionsPerPacket_)
        && configReader.lookupValue("inputBufferSize", bufferSize_)
        && configReader.lookupValue("numberOfDetectorModules", numberOfDetectorModules_)) {
     if(transportProt == "udp")
        transportProtocol_ = transportProtocol::UDP;
     else if(transportProt == "tcp")
        transportProtocol_ = transportProtocol::TCP;
     else
        transportProtocol_ = transportProtocol::UDP;
     numberOfProjections_ = samplingRate * 1000000 / scanRate;
     return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

}
