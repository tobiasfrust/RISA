/*
 * Copyright 2016
 *
 * Loader.h
 *
 *  Created on: 13.06.2016
 *      Author: Tobias Frust
 */

#ifndef LOADER_H_
#define LOADER_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <queue>

#include <tiffio.h>

#include "../Basics/performance.h"

#include <boost/log/trivial.hpp>

#include "ddrf/Image.h"
#include "ddrf/MemoryPool.h"
#include "ddrf/cuda/HostMemoryManager.h"

namespace risa
{
      class OfflineLoader
      {
         public:
            using manager_type = ddrf::cuda::HostMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

         public:
            OfflineLoader(const std::string& configFile);

            auto loadImage() -> ddrf::Image<manager_type>;

         protected:
            ~OfflineLoader();


         private:
            unsigned int memoryPoolIndex_;
            std::queue<ddrf::Image<manager_type>> buffer_;

            double worstCaseTime_;
            double bestCaseTime_;
            Timer tmr_;

            //configuration parameters
            std::string path_, fileName_, fileEnding_;
            int numberOfDetectors_, numberOfProjections_;
            int numberOfDetectorModules_, numberOfPlanes_;
            unsigned int numberOfFrames_;

            std::size_t stopFrame_;
            std::size_t index_;

            auto readInput() -> void;
            auto readConfig(const std::string& configFile) -> bool;
      };
   }


#endif /* LOADER_H_ */
