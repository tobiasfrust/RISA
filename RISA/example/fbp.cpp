/*
 * Copyright 2016
 *
 * Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#include <risa/Filter/Filter.h>
#include <risa/Backprojection/Backprojection.h>
#include <risa/Copy/D2H.h>
#include <risa/Copy/H2D.h>
#include <risa/Loader/OfflineLoader.h>
#include <risa/Loader/TiffLoader.h>
#include <risa/Saver/OfflineSaver.h>
#include <risa/Receiver/Receiver.h>
#include <risa/Reordering/Reordering.h>

#include <ddrf/Image.h>
#include <ddrf/ImageLoader.h>
#include <ddrf/ImageSaver.h>
#include <ddrf/imageLoaders/TIFF/TIFF.h>
#include <ddrf/imageSavers/TIFF/TIFF.h>

#include <ddrf/pipeline/Pipeline.h>
#include <ddrf/pipeline/SinkStage.h>
#include <ddrf/pipeline/SourceStage.h>
#include <ddrf/pipeline/Stage.h>

#include <ddrf/cuda/HostMemoryManager.h>

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

   using tiffLoader = ddrf::ImageLoader<risa::loaders::TIFF<float>>;
   //using offlineLoader = ddrf::ImageLoader<risa::OfflineLoader>;
   //using onlineReceiver = ddrf::ImageLoader<risa::Receiver>;
   //using tiffSaver = ddrf::ImageSaver<ddrf::savers::TIFF<ddrf::cuda::HostMemoryManager<float, ddrf::cuda::async_copy_policy>>>;
   using offlineSaver = ddrf::ImageSaver<risa::OfflineSaver>;

   using sourceStage = ddrf::pipeline::SourceStage<tiffLoader>;
   using copyStageH2D = ddrf::pipeline::Stage<risa::cuda::H2D>;
   using filterStage = ddrf::pipeline::Stage<risa::cuda::Filter>;
   using backProjectionStage = ddrf::pipeline::Stage<risa::cuda::Backprojection>;
   using maskingStage = ddrf::pipeline::Stage<risa::cuda::Masking>;
   using copyStageD2H = ddrf::pipeline::Stage<risa::cuda::D2H>;
   using sinkStage = ddrf::pipeline::SinkStage<offlineSaver>;

   int numberofDevices;
   CHECK(cudaGetDeviceCount(&numberofDevices));

   try {
      //set up pipeline
      auto pipeline = ddrf::pipeline::Pipeline { };

      auto source = pipeline.create<sourceStage>(address, configFile);
      auto h2d = pipeline.create<copyStageH2D>(configFile);
      auto filter = pipeline.create<filterStage>(configFile);
      auto backProjection = pipeline.create<backProjectionStage>(configFile);
      //Falls etwas maskiert werden soll, einkommentieren und die Maske den Wünschen nach anpassen
      //auto masking = pipeline.create<maskingStage>(configFile);
      auto d2h = pipeline.create<copyStageD2H>(configFile);
      auto sink = pipeline.create<sinkStage>(outputPath, prefix, configFile);

      pipeline.connect(source, h2d);
      pipeline.connect(h2d, filter);
      pipeline.connect(filter, backProjection);
      pipeline.connect(backProjection, d2h);
      pipeline.connect(d2h, sink);

      pipeline.run(source, h2d, filter, backProjection, d2h, sink);

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
