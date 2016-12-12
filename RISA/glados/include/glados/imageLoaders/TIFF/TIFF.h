/*
 * This file is part of the GLADOS-library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * GLADOS is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GLADOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with GLADOS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 30 November 2016
 * Authors: Tobias Frust <t.frust@hzdr.de>
 *
 */

#ifndef GLADOS_TIFF_L_H_
#define GLADOS_TIFF_L_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <tiffio.h>

#include <boost/log/trivial.hpp>

#include "../../Image.h"
#include "../../MemoryPool.h"

namespace glados
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
				// TODO: Implement support for more than one frame per file
				TIFF(){
					memoryPoolIndex_ = MemoryPool<MemoryManager>::instance()->registerStage(40, 432*500);
				}

				auto loadImage(const std::string& path, std::size_t index) -> Image<MemoryManager>
				{
					using empty_return = Image<MemoryManager>;

					auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter>{TIFFOpen(path.c_str(), "rb")};
					BOOST_LOG_TRIVIAL(debug) << "glados::loaders::TIFF: Open file " << path << " for reading.";
					if(tif == nullptr)
						throw std::runtime_error{"glados::loaders::TIFF: Could not open file " + path + " for reading."};

					int imageWidth, imageLength;

					TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imageLength);
					TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &imageWidth);
					// read image data
				   auto img = MemoryPool<MemoryManager>::instance()->requestMemory(memoryPoolIndex_);

					for(auto row = 0; row < imageLength; row++){
						if(TIFFReadScanline(tif.get(), img.container().get() + row * imageWidth, row, 0) != 1){
						   throw std::runtime_error{"glados::loaders::TIFF: Could not read scanline."};
						}
					}
					img.setIdx(index);
					img.setPlane(0);

					BOOST_LOG_TRIVIAL(debug) << "glados::loaders::TIFF: Sent file " << path << ".";

					return std::move(img);
				}

			protected:
				~TIFF(){
				   glados::MemoryPool<MemoryManager>::instance()->freeMemory(memoryPoolIndex_);
					BOOST_LOG_TRIVIAL(info) << "glados::loaders::detail::TIFF: Destroyed";
				}


			private:
				unsigned int memoryPoolIndex_;
		};
	}
}


#endif /* GLADOS_TIFF_L_ */
