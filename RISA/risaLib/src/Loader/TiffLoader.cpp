#include <risa/Loader/TiffLoader.h>
#include <risa/ConfigReader/ConfigReader.h>

#include <ddrf/Filesystem.h>

namespace risa {
namespace loaders {

TIFF::TIFF(const std::string& address, const std::string& configFile){
   if (readConfig(configFile)) {
      throw std::runtime_error(
            "recoLib::OfflineLoader: Configuration file could not be loaded successfully. Please check!");
   }

   paths_ = ddrf::readDirectory(inputPath_);

   memoryPoolIndex_ = ddrf::MemoryPool<manager_type>::instance()->registerStage(40, numberOfDetectors_*numberOfProjections_);
}

TIFF::~TIFF(){
   ddrf::MemoryPool<manager_type>::instance()->freeMemory(memoryPoolIndex_);
  					BOOST_LOG_TRIVIAL(info) << "ddrf::loaders::detail::TIFF: Destroyed";
}

auto TIFF::loadImage() -> ddrf::Image<manager_type>{
	using empty_return = ddrf::Image<manager_type>;

	if(paths_.size() <= index_)
		return ddrf::Image<manager_type>();

	auto path = paths_[index_];

	auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter>{TIFFOpen(path.c_str(), "rb")};
	BOOST_LOG_TRIVIAL(debug) << "ddrf::loaders::TIFF: Open file " << path << " for reading.";
	if(tif == nullptr)
		throw std::runtime_error{"ddrf::loaders::TIFF: Could not open file " + path + " for reading."};

	int imageWidth, imageLength;

	TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imageLength);
	TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &imageWidth);

   if(imageWidth != numberOfDetectors_ || imageLength != numberOfProjections_){
      throw std::runtime_error{"risa:loader:tiff: file has wrong input size: " + path};
   }

	// read image data
   auto img = ddrf::MemoryPool<manager_type>::instance()->requestMemory(memoryPoolIndex_);

	for(auto row = 0; row < imageLength; row++){
		if(TIFFReadScanline(tif.get(), img.container().get() + row * imageWidth, row, 0) != 1){
		   throw std::runtime_error{"ddrf::loaders::TIFF: Could not read scanline."};
		}
	}
	img.setIdx(index_);
	index_++;
	img.setPlane(0);

	BOOST_LOG_TRIVIAL(debug) << "ddrf::loaders::TIFF: Sent file " << path << ".";

	return std::move(img);
}

auto TIFF::readConfig(const std::string& configFile) -> bool{
   ConfigReader configReader = ConfigReader(configFile.data());
   if (configReader.lookupValue("numberOfParallelDetectors", numberOfDetectors_)
         && configReader.lookupValue("dataInputPath", inputPath_)
         && configReader.lookupValue("numberOfParallelProjections", numberOfProjections_)) {
      return EXIT_SUCCESS;
   }
   return EXIT_FAILURE;
}

}
}
