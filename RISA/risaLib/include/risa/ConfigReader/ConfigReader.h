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

   //!   Reads the configuration values from the input file.
   /**
    * This class takes the file path to a configuration file as input and reads it
    * using the libconfig++-Library
    */
class ConfigReader {

public:

   //! Create the configuration reader from the given input file
   /**
    *
    * @param[in]  configFile  path to configuration file
    *
    */
   ConfigReader(const char* configFile);

   ConfigReader(const ConfigReader& configReader) {}

   //! Reads the configuration value identified by the identifier-string
   /**
    * @param[in]  identifier  the string used to identify the desired parameter in the configuration file
    * @param[out] value       the value, that was passed in the configuration file
    * @retval true   the parameter could be read successfully
    * @retval false  the parameter could not be read successfully
    */
   template<typename T>
   bool lookupValue(const std::string& identifier, T& value) {
	  bool ret = cfg.lookupValue(identifier.c_str(), value);
	  BOOST_LOG_TRIVIAL(debug) << "Configuration value " << identifier << ": " << value;
     return ret;
   }

   //! Reads the configuration value list identified by the identifier-string and returns the value stored at position index
   /**
    * @param[in]  identifier  the string used to identify the desired parameter in the configuration file
    * @param[in]  index       the position at which the desired value shall be read
    * @param[out] value       the value, that was passed in the configuration file
    * @retval true   the parameter could be read successfully
    * @retval false  the parameter could not be read successfully
    */
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
};
}

#endif
