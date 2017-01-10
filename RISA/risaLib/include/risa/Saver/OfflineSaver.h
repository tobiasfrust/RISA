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

#ifndef OFFLINESAVER_H_
#define OFFLINESAVER_H_

#include "../Basics/performance.h"
#include "../ConfigReader/read_json.hpp"

#include <glados/cuda/HostMemoryManager.h>
#include <glados/Image.h>
#include <glados/CircularBuffer.h>

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
      using manager_type = glados::cuda::HostMemoryManager<float, glados::cuda::async_copy_policy>;

   public:
      OfflineSaver(const std::string& config_file);

      //!< this function is called, when an image exits the software pipeline
      auto saveImage(glados::Image<manager_type> image, std::string path) -> void;

   protected:
      ~OfflineSaver();

   private:
      //! Creates a Tiff Sequence to be stored on disk
      /**
       * @param[in] planeID   specifies, which buffer is to be stored on hard disk
       */
      auto writeTiffSequence(const int planeID) -> void;
      auto readConfig(const read_json& config_reader) -> bool;
      //! writes a single image to the tiff sequence
      /**
       * @param[in]  tif   pointer to the TIFF-sequence
       * @param[in]  img   the image to be written into the tiff-file
       */
      auto writeToTiff(::TIFF* tif, glados::Image<manager_type> img) const -> void;

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
      std::vector<glados::CircularBuffer<glados::Image<manager_type>>> outputBuffers_;
      //std::vector<std::vector<glados::Image<manager_type>>> outputBuffers_;

   };

}

#endif /* OFFLINESAVER_H_ */
