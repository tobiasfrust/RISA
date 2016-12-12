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
 * Authors: Tobias Frust (FWCC) <t.frust@hzdr.de>
 *
 */

#include <risa/Saver/OfflineSaver.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <glados/Image.h>
#include <glados/MemoryPool.h>

#include <boost/log/trivial.hpp>

#include <tiffio.h>

#include <exception>
#include <string>
#include <iostream>

namespace risa {

namespace detail {
template<class T, bool = std::is_integral<T>::value, bool =
      std::is_unsigned<T>::value> struct SampleFormat {
};
template<class T> struct SampleFormat<T, true, true> {
   static constexpr auto value = SAMPLEFORMAT_UINT;
};
template<class T> struct SampleFormat<T, true, false> {
   static constexpr auto value = SAMPLEFORMAT_INT;
};
template<> struct SampleFormat<float> {
   static constexpr auto value = SAMPLEFORMAT_IEEEFP;
};
template<> struct SampleFormat<double> {
   static constexpr auto value = SAMPLEFORMAT_IEEEFP;
};

template<class T> struct BitsPerSample {
   static constexpr auto value = sizeof(T) << 3;
};

struct TIFFDeleter {
   auto operator()(::TIFF* p) -> void {
      TIFFClose(p);
   }
};
}

OfflineSaver::OfflineSaver(const std::string& configFile) {
   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::OfflineSaver: Configuration file could not be loaded successfully. Please check!");
   }

   for (auto i = 0; i < numberOfPlanes_; i++) {
      outputBuffers_.emplace_back(circularBufferSize_);
      fileIndex_.push_back(0u);
      //outputBuffers_[i].reserve(numberOfFrames_);
   }

   memoryPoolIndex_ = glados::MemoryPool<manager_type>::instance()->registerStage(
         numberOfPlanes_ * (circularBufferSize_+1),
         numberOfPixels_ * numberOfPixels_);

//   memoryPoolIndex_ = glados::MemoryPool<manager_type>::instance()->registerStage(
//         numberOfPlanes_ * (circularBufferSize_+1),
//         256*1024);

   for(auto i = 0; i < numberOfPlanes_; i++){
      tmrs_.emplace_back();
      tmrs_[i].start();
   }
}

OfflineSaver::~OfflineSaver() {
   for(auto i = 0; i < numberOfPlanes_; i++){
      writeTiffSequence(i);
   }
   BOOST_LOG_TRIVIAL(info) << "Maximum latency: " << maxLatency_ << " ms; minimum latency: " << minLatency_ << " ms";
   glados::MemoryPool<manager_type>::instance()->freeMemory(memoryPoolIndex_);
}

auto OfflineSaver::saveImage(glados::Image<manager_type> image,
      std::string path) -> void {
   auto img = glados::MemoryPool<manager_type>::instance()->requestMemory(
         memoryPoolIndex_);
   std::copy(image.container().get(),
         image.container().get() + image.size(),
         img.container().get());
   img.setIdx(image.index());
   img.setPlane(image.plane());
   img.setStart(image.start());
   auto latency = img.duration();
   if(latency < minLatency_)
      minLatency_ = latency;
   else if(latency > maxLatency_)
      maxLatency_ = latency;
   outputBuffers_[image.plane()].push_back(std::move(img));
   if(mode_ == detail::RecoMode::offline){
      if(outputBuffers_[image.plane()].full()){
         writeTiffSequence(image.plane());
      }
   }else if(mode_ == detail::RecoMode::online){
      tmrs_[image.plane()].stop();
      double diff = tmrs_[image.plane()].elapsed();
      if(diff < 0.02)
         return;
      std::string path = outputPath_ + fileName_ + "_plane"
               + std::to_string(image.plane()) + "_" + "0" + ".tif";
      auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter> { TIFFOpen(
            path.c_str(), "wb") };
      if (tif == nullptr)
         throw std::runtime_error { "recoLib::OfflineSaver: Could not open file "
               + path + " for writing." };
      writeToTiff(tif.get(), std::move(image));
      tmrs_[image.plane()].start();
   }
}

auto OfflineSaver::writeTiffSequence(const int planeID) -> void {
   if(outputBuffers_[planeID].count() == 0)
      return;
   std::string path = outputPath_ + fileName_ + "_plane"
         + std::to_string(planeID) + "_" + std::to_string(fileIndex_[planeID]) + "seq.tif";
   auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter> { TIFFOpen(
         path.c_str(), "w8") };
   if (tif == nullptr)
      throw std::runtime_error { "recoLib::OfflineSaver: Could not open file "
            + path + " for writing." };

   for(auto i = 0u; i < outputBuffers_[planeID].count(); i++){
      BOOST_LOG_TRIVIAL(debug) << "writing out index " << i;
      writeToTiff(tif.get(), std::move(outputBuffers_[planeID].at(i)));
      if(TIFFWriteDirectory(tif.get()) != 1)
         throw std::runtime_error{"savers::TIFF: tiffio error while writing to " + path};
   }
   ++fileIndex_[planeID];
   outputBuffers_[planeID].clear();
}

auto OfflineSaver::writeToTiff(::TIFF* tif, glados::Image<manager_type> img) const -> void {

   TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, numberOfPixels_);
   TIFFSetField(tif, TIFFTAG_IMAGELENGTH, numberOfPixels_);
   TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
   TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
   TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
   TIFFSetField(tif, TIFFTAG_THRESHHOLDING, THRESHHOLD_BILEVEL);
   TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
   TIFFSetField(tif, TIFFTAG_SOFTWARE, "RISA");
   TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

   auto data = img.data();
   auto dataPtr = data;
   for (auto row = 0; row < numberOfPixels_; ++row) {
      TIFFWriteScanline(tif, dataPtr, row);
      dataPtr += numberOfPixels_;
   }
}

/**
 * All values needed for setting up the class are read from the config file
 * in this function.
 *
 * @param[in] configFile path to config file
 *
 * @return returns true, if configuration file could be read successfully, else false
 */
auto OfflineSaver::readConfig(const std::string& configFile) -> bool {
   ConfigReader configReader = ConfigReader(configFile.data());
   std::string mode;
   if (configReader.lookupValue("numberOfPixels", numberOfPixels_)
         && configReader.lookupValue("outputPath", outputPath_)
         && configReader.lookupValue("outputFileName", fileName_)
         && configReader.lookupValue("numberOfDataFrames", numberOfFrames_)
         && configReader.lookupValue("numberOfPlanes", numberOfPlanes_)
         && configReader.lookupValue("outputBufferSize", circularBufferSize_)
         && configReader.lookupValue("mode", mode)) {
      if(mode == "offline"){
         mode_ = detail::RecoMode::offline;
      }else if(mode == "online"){
         mode_ = detail::RecoMode::online;
      }else{
         BOOST_LOG_TRIVIAL(error) << "Requested mode \"" << mode << "\" not supported.";
         return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;
}

}
