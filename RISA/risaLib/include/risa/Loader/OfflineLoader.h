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
#ifndef LOADER_H_
#define LOADER_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <type_traits>
#include <utility>
#include <queue>

#include <tiffio.h>

#include "../Basics/performance.h"

#include <boost/log/trivial.hpp>

#include "glados/Image.h"
#include "glados/MemoryPool.h"
#include "glados/cuda/HostMemoryManager.h"

namespace risa
{
//! The loader stage for offline data processing.
/**
 * This stage gets the paths to the input files and loads data sinogram by sinogram. Hence, the
 * the lower limit of the reconstruction speed is given by the bandwidth of the storage system.
 * Furthermore, this method leads to low memory consumption.
 */
      class OfflineLoader
      {
         public:
            using manager_type = glados::cuda::HostMemoryManager<unsigned short, glados::cuda::async_copy_policy>;

         public:
            OfflineLoader(const std::string& address, const std::string& configFile);

            //! #loadImage is called, when the software pipeline is able to process a new image
            /**
             * @return  the image that is pushed through the software pipeline
             */
            auto loadImage() -> glados::Image<manager_type>;

         protected:
            ~OfflineLoader();


         private:
            unsigned int memoryPoolIndex_;    //!<  stores the indeces received when registering in MemoryPool
            std::queue<glados::Image<manager_type>> buffer_;

            double worstCaseTime_;
            double bestCaseTime_;
            Timer tmr_;

            //configuration parameters
            std::string path_;      //!< the input path of raw data
            std::string fileName_;  //!< the input filename
            std::string fileEnding_;   //!< the fileending of the input data
            int numberOfDetectors_; //!< the number of detectors in the fan beam sinogram
            int numberOfProjections_;  //!< the number of projections in the fan beam sinogram
            int numberOfDetectorModules_; //!< the number of detector modules
            int numberOfPlanes_; //!< the number of planes
            unsigned int numberOfFrames_; //!< the number of frames in the input data for one plane
            int numberOfDetectorsPerModule_; //!< the number of detectors per module

            std::size_t stopFrame_;
            std::size_t index_{0u};

            std::vector<std::unique_ptr<std::ifstream>> ifstreams_; //!< stores the input file streams during the lifetime of the stage

            auto readInput() -> void;
            auto readConfig(const std::string& configFile) -> bool;
      };
   }


#endif /* LOADER_H_ */
