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

   //! This stage is suited for online and offline processing via configuration options.
   /**
    * In offline mode it has numberOfPlanes output buffers of fixed size. If a buffer is full, it is written to disk
    * and cleared afterwars. In online mode, the output acts as a circular buffer. The oldest values are
    * overwritten, if the buffer is full. At program exit, the buffer is written to disk
    */
   class OfflineSaver {
   public:
      using manager_type = ddrf::cuda::HostMemoryManager<float, ddrf::cuda::async_copy_policy>;

   public:
      OfflineSaver(const std::string& configFile);

      //!< this function is called, when an image exits the software pipeline
      auto saveImage(ddrf::Image<manager_type> image, std::string path) -> void;

   protected:
      ~OfflineSaver();

   private:
      //! Creates a Tiff Sequence to be stored on disk
      /**
       * @param[in] planeID   specifies, which buffer is to be stored on hard disk
       */
      auto writeTiffSequence(const int planeID) -> void;
      auto readConfig(const std::string& configFile) -> bool;
      //! writes a single image to the tiff sequence
      /**
       * @param[in]  tif   pointer to the TIFF-sequence
       * @param[in]  img   the image to be written into the tiff-file
       */
      auto writeToTiff(::TIFF* tif, ddrf::Image<manager_type> img) const -> void;

      int memoryPoolIndex_;

      int numberOfPixels_; //!< the number of pixels in the reconstructed image in one dimension
      int numberOfFrames_;
      int numberOfPlanes_; //!< the number of planes
      int framesPerFile_;

      detail::RecoMode mode_;

      unsigned int circularBufferSize_;   //!< the size of the output buffers

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
