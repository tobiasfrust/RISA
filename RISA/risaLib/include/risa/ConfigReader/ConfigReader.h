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
