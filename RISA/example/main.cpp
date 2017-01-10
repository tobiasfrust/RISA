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
 */

#include <risa/ConfigReader/read_json.hpp>
#include <risa/Filter/Filter.h>
#include <risa/Backprojection/Backprojection.h>
#include <risa/Attenuation/Attenuation.h>
#include <risa/Copy/D2H.h>
#include <risa/Copy/H2D.h>
#include <risa/Fan2Para/Fan2Para.h>
#include <risa/Masking/Masking.h>
#include <risa/Loader/OfflineLoader.h>
#include <risa/Loader/OfflineLoader_perfTest.h>
#include <risa/Saver/OfflineSaver.h>
#include <risa/Receiver/Receiver.h>
#include <risa/Reordering/Reordering.h>

#include <glados/Image.h>
#include <glados/ImageLoader.h>
#include <glados/ImageSaver.h>
#include <glados/imageLoaders/TIFF/TIFF.h>
#include <glados/imageSavers/TIFF/TIFF.h>

#include <glados/pipeline/Pipeline.h>
#include <glados/pipeline/SinkStage.h>
#include <glados/pipeline/SourceStage.h>
#include <glados/pipeline/Stage.h>

#include <glados/cuda/HostMemoryManager.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include <cuda_profiler_api.h>

void initLog() {
#ifndef NDEBUG
   boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
#else
   boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
#endif
}

int main(int argc, char *argv[]) {

   //nvtxNameOsThreadA(pthread_self(), "Main");

   initLog();

   if(argc < 2){
      BOOST_LOG_TRIVIAL(error) << "Call the program like this: ./example <path_to_config_file>";
      return EXIT_FAILURE;
   }
   std::string configFile = argv[1];

   //TODO: read output path from config
   auto outputPath = std::string { "Reco" };
   auto inputPath = std::string { "Sino" };
   auto prefix = std::string { "IMG" };
   //auto configFile = std::string { "config.cfg" };
   auto address = std::string { "10.0.0.10" };

   //using tiffLoader = glados::ImageLoader<glados::loaders::TIFF<glados::cuda::HostMemoryManager<unsigned short, glados::cuda::async_copy_policy>>>;
   using offlineLoader = glados::ImageLoader<risa::OfflineLoader>;
   using onlineReceiver = glados::ImageLoader<risa::Receiver>;
   //using tiffSaver = glados::ImageSaver<glados::savers::TIFF<glados::cuda::HostMemoryManager<float, glados::cuda::async_copy_policy>>>;
   using offlineSaver = glados::ImageSaver<risa::OfflineSaver>;

   using sourceStage = glados::pipeline::SourceStage<offlineLoader>;
   using copyStageH2D = glados::pipeline::Stage<risa::cuda::H2D>;
   using reorderingStage = glados::pipeline::Stage<risa::cuda::Reordering>;
   using attenuationStage = glados::pipeline::Stage<risa::cuda::Attenuation>;
   using fan2ParaStage = glados::pipeline::Stage<risa::cuda::Fan2Para>;
   using filterStage = glados::pipeline::Stage<risa::cuda::Filter>;
   using backProjectionStage = glados::pipeline::Stage<risa::cuda::Backprojection>;
   using maskingStage = glados::pipeline::Stage<risa::cuda::Masking>;
   using copyStageD2H = glados::pipeline::Stage<risa::cuda::D2H>;
   using sinkStage = glados::pipeline::SinkStage<offlineSaver>;

   int numberofDevices;
   CHECK(cudaGetDeviceCount(&numberofDevices));

   try {
      //set up pipeline
      auto pipeline = glados::pipeline::Pipeline { };

      auto h2d = pipeline.create<copyStageH2D>(configFile);
      auto reordering = pipeline.create<reorderingStage>(configFile);
      auto attenuation = pipeline.create<attenuationStage>(configFile);
      auto fan2Para = pipeline.create<fan2ParaStage>(configFile);
      auto filter = pipeline.create<filterStage>(configFile);
      auto backProjection = pipeline.create<backProjectionStage>(configFile);
      //auto masking = pipeline.create<maskingStage>(configFile);
      auto d2h = pipeline.create<copyStageD2H>(configFile);
      auto sink = pipeline.create<sinkStage>(outputPath, prefix, configFile);
      auto source = pipeline.create<sourceStage>(address, configFile);

      pipeline.connect(source, h2d);
      pipeline.connect(h2d, reordering);
      pipeline.connect(reordering, attenuation);
      pipeline.connect(attenuation, fan2Para);
      pipeline.connect(fan2Para, filter);
      pipeline.connect(filter, backProjection);
      //pipeline.connect(backProjection, masking);
      pipeline.connect(backProjection, d2h);
      pipeline.connect(d2h, sink);

      pipeline.run(source, h2d, reordering, attenuation, fan2Para, filter, backProjection, d2h, sink);
      BOOST_LOG_TRIVIAL(info) << "Initialization finished.";

      for (auto i = 0; i < numberofDevices; i++){
         CHECK(cudaSetDevice(i));
         CHECK(cudaProfilerStart());
      }

      pipeline.wait();

      for (auto i = 0; i < numberofDevices; i++){
         CHECK(cudaSetDevice(i));
         CHECK(cudaProfilerStop());
      }

   } catch (const std::runtime_error& err) {
      std::cerr << "=========================" << std::endl;
      std::cerr << "A runtime error occurred: " << std::endl;
      std::cerr << err.what() << std::endl;
      std::cerr << "=========================" << std::endl;
   }

   for(auto i = 0; i < numberofDevices; i++){
      CHECK(cudaSetDevice(i));
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaDeviceReset());
   }
   return 0;
}
