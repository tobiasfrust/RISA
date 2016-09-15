#ifndef RISA_TIFF_L_H_
#define RISA_TIFF_L_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <tiffio.h>

#include <boost/log/trivial.hpp>

#include "../../Image.h"
#include "../../MemoryPool.h"

namespace risa
{
	namespace loaders
	{
		namespace detail
		{
			template<class T, bool = std::is_integral<T>::value, bool = std::is_unsigned<T>::value> struct SampleFormat {};
			template<class T> struct SampleFormat<T, true, true> { static constexpr auto value = SAMPLEFORMAT_UINT; };
			template<class T> struct SampleFormat<T, true, false> { static constexpr auto value = SAMPLEFORMAT_INT; };
			template<> struct SampleFormat<float> { static constexpr auto value = SAMPLEFORMAT_IEEEFP; };
			template<> struct SampleFormat<double>{ static constexpr auto value = SAMPLEFORMAT_IEEEFP; };

			template<class T> struct BitsPerSample { static constexpr auto value = sizeof(T) << 3; };

			struct TIFFDeleter { auto operator()(TIFF* p) -> void { TIFFClose(p); }};
		}

      template <class MemoryManager>
		class TIFF : public MemoryManager
		{
			public:
            using value_type = typename MemoryManager::value_type;
				using manager_type = MemoryManager;

			public:
				TIFF(const std::string& address, const std::string& configFile){
               if (readConfig(configFile)) {
                  throw std::runtime_error(
                        "recoLib::OfflineLoader: Configuration file could not be loaded successfully. Please check!");
               }

					memoryPoolIndex_ = MemoryPool<MemoryManager>::instance()->registerStage(40, numberOfDetectors*numberOfProjections);
				}

				auto loadImage() -> Image<MemoryManager>
				{
					using empty_return = Image<MemoryManager>;

					auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter>{TIFFOpen(path.c_str(), "rb")};
					BOOST_LOG_TRIVIAL(debug) << "ddrf::loaders::TIFF: Open file " << path << " for reading.";
					if(tif == nullptr)
						throw std::runtime_error{"ddrf::loaders::TIFF: Could not open file " + path + " for reading."};

					int imageWidth, imageLength;

					TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imageLength);
					TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &imageWidth);

               if(imageWidth != numberOfDetectors_ || imageLength != numberOfProjections_){
                  throw std::runtime_error{"file has wrong input size: " + path};
                  return
               }

					// read image data
				   auto img = MemoryPool<MemoryManager>::instance()->requestMemory(memoryPoolIndex_);

					for(auto row = 0; row < imageLength; row++){
						if(TIFFReadScanline(tif.get(), img.container().get() + row * imageWidth, row, 0) != 1){
						   throw std::runtime_error{"ddrf::loaders::TIFF: Could not read scanline."};
						}
					}
					img.setIdx(index);
					img.setPlane(0);

					BOOST_LOG_TRIVIAL(debug) << "ddrf::loaders::TIFF: Sent file " << path << ".";

					return std::move(img);
				}

			protected:
				~TIFF(){
				   ddrf::MemoryPool<MemoryManager>::instance()->freeMemory(memoryPoolIndex_);
					BOOST_LOG_TRIVIAL(info) << "ddrf::loaders::detail::TIFF: Destroyed";
				}


			private:
            auto readConfig(const std::string& configFile) -> bool {
               ConfigReader configReader = ConfigReader(configFile.data());
               int samplingRate, scanRate;
               if (configReader.lookupValue("numberOfParallelDetectors", numberOfDetectors_)
                     && configReader.lookupValue("dataInputPath", inputPath_)
                     && configReader.lookupValue("numberOfParallelProjections", numberOfProjections_)) {
                  return EXIT_SUCCESS;
               }
               return EXIT_FAILURE;
            }


				unsigned int memoryPoolIndex_;
            std::string inputPath_;
            int numberOfDetectors_;
            int numberOfProjections_;
		};
	}
}


#endif /* DDRF_TIFF_L_ */
