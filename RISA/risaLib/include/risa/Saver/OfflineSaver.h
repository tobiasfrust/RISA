/*
 * Copyright 2016
 *
 * OfflineSaver.h
 *
 *  Created on: 15.06.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef OFFLINESAVER_H_
#define OFFLINESAVER_H_

#include "../Basics/performance.h"

#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/Image.h>
#include <ddrf/CircularBuffer.h>

#include <tiffio.h>

//#include <boost/circular_buffer.hpp>

#include <vector>
#include <string>

namespace risa{

   namespace detail{
      enum RecoMode: short {
         offline,
         online
      };
   }

   class OfflineSaver {
   public:
      using manager_type = ddrf::cuda::HostMemoryManager<float, ddrf::cuda::async_copy_policy>;

   public:
      OfflineSaver(const std::string& configFile);

      auto saveImage(ddrf::Image<manager_type> image, std::string path) -> void;

   protected:
      ~OfflineSaver();

   private:
      auto writeTiffSequence(const int planeID) -> void;
      auto readConfig(const std::string& configFile) -> bool;
      auto writeToTiff(::TIFF* tif, ddrf::Image<manager_type> img) const -> void;

      int memoryPoolIndex_;

      int numberOfPixels_, numberOfFrames_;
      int numberOfPlanes_, framesPerFile_;

      detail::RecoMode mode_;

      unsigned int circularBufferSize_;

      std::string outputPath_, fileName_;

      double minLatency_{std::numeric_limits<double>::max()};
      double maxLatency_{0.0};

      std::vector<Timer> tmrs_;

      std::vector<std::size_t> fileIndex_;
      std::vector<ddrf::CircularBuffer<ddrf::Image<manager_type>>> outputBuffers_;
      //std::vector<std::vector<ddrf::Image<manager_type>>> outputBuffers_;

   };

}

#endif /* OFFLINESAVER_H_ */
