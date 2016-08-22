/*
 *  Copyright 2016
 *
 *  ConfigReader.cpp
 *
 *  Created on: 18.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <risa/ConfigReader/ConfigReader.h>

namespace risa {
  
ConfigReader::ConfigReader(const char* configFile) {
   try {
      cfg.readFile(configFile);
   } catch (const libconfig::FileIOException &fioex) {
      std::cerr << "I/O error while reading file." << std::endl;
      exit(1);
   } catch (const libconfig::ParseException &pex) {
      std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
            << " - " << pex.getError() << std::endl;
      exit(1);
   }
}
}
