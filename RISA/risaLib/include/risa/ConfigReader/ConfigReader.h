/*
 * Copyright 2016
 *
 *  ConfigReader.h
 *
 *  Created on: 18.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef CONFIGREADER_H
#define CONFIGREADER_H
#pragma once

#include <boost/log/trivial.hpp>

#include <libconfig.h++>

#include <string>

namespace risa {
class ConfigReader {
public:
   ConfigReader(const char* configFile);
   ConfigReader(const ConfigReader& configReader) {
   }

   template<typename T>
   bool lookupValue(const std::string& identifier, T& value) {
	  bool ret = cfg.lookupValue(identifier.c_str(), value);
	  BOOST_LOG_TRIVIAL(debug) << "Configuration value " << identifier << ": " << value;
      return ret;
   }

   template<typename T>
   bool lookupValue(const std::string& identifier, int index, T& value) {
	   libconfig::Setting& s = cfg.lookup(identifier.c_str());
	   if(s.getLength() > index){
		   value = s[index];
		   BOOST_LOG_TRIVIAL(debug) << "Configuration value " << identifier << "[" << index << "]: " << value;
		   return true;
	   }
	   return false;
   }

private:
   libconfig::Config cfg;
   float cudaTest();
};
}

#endif
