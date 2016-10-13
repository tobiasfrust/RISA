/*
 * Copyright 2016
 *
 * Loader.h
 *
 *  Created on: 13.06.2016
 *      Author: Tobias Frust
 */

#ifndef LOADER_OFFLINE_PERFTEST_H_
#define LOADER_OFFLINE_PERFTEST_H_

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
//! This stage fills the software pipeline with input data stored in ram.
/**
 * To perform a longer performance test, input data is sent out continously up to a specified number of frames.
 *
 */
      class OfflineLoaderPerfTest
      {
         public:
            using manager_type = ddrf::cuda::HostMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

         public:
            OfflineLoaderPerfTest(const std::string& address, const std::string& configFile);

            //! #loadImage is called, when the software pipeline is able to process a new image
            /**
             * @return  the image that is pushed through the software pipeline
             */
            auto loadImage() -> ddrf::Image<manager_type>;

         protected:
            ~OfflineLoaderPerfTest();


         private:
            unsigned int memoryPoolIndex_;   //!<  stores the indeces received when registering in MemoryPool
            std::queue<ddrf::Image<manager_type>> buffer_;  //!<  the buffer which stores the test data set

            double worstCaseTime_;
            double bestCaseTime_;
            Timer tmr_;

            //configuration parameters
            std::string path_;      //!< the input path of raw data
            std::string fileName_;  //!< the input filename
            std::string fileEnding_;   //!< the fileending of the input data
            int numberOfDetectors_; //!< the number of detectors in the fan beam sinogram
            int numberOfProjections_;   //!< the number of projections in the fan beam sinogram
            int numberOfDetectorModules_; //!< the number of detector modules
            int numberOfPlanes_; //!< the number of planes
            unsigned int numberOfFrames_; //!< the number of frames in the input data for one plane

            std::size_t stopFrame_;
            std::size_t index_;

            auto readInput() -> void;
            auto readConfig(const std::string& configFile) -> bool;
      };
   }


#endif /* LOADER_OFFLINE_PERFTEST_H_ */
