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

#ifndef SAVERS_TIFF_H_
#define SAVERS_TIFF_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <time.h>

#include <tiffio.h>

#include "../../Image.h"
#include "../../Volume.h"
#include "../../MemoryPool.h"

namespace ddrf {
namespace savers {
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
	auto operator()(TIFF* p) -> void {
		TIFFClose(p);
	}
};
}

template<class MemoryManager>
class TIFF: public MemoryManager {
public:
	using manager_type = MemoryManager;

public:

	auto saveImage(Image<MemoryManager> image,
			std::string& path) const -> void {
		using value_type = typename Image<MemoryManager>::value_type;
		if(image.plane() == 0)
		   return;

		unsigned int numberOfPixels = sqrt(image.size());

		path.append(".tif");
		auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter> { TIFFOpen(
				path.c_str(), "w") };
		if (tif == nullptr)
			throw std::runtime_error { "savers::TIFF: Could not open file "
					+ path + " for writing." };

		TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, numberOfPixels);
		TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, numberOfPixels);
		TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE,
				detail::BitsPerSample<value_type>::value);
		TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT,
				detail::SampleFormat<value_type>::value);
		TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

		auto data = image.data();
		auto dataPtr = data;

		for (auto row = 0u; row < numberOfPixels; ++row) {
			if (TIFFWriteScanline(tif.get(), dataPtr + row * numberOfPixels,
					row) != 1) {
				throw std::runtime_error {
						"savers::TIFF: tiffio error while writing scanline " };
			}
		}
	}

	auto saveVolume(Volume<MemoryManager> volume,
			std::string& path) const -> void {
		using value_type = typename Volume<MemoryManager>::value_type;
		path.append(".tif");

		auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter> { TIFFOpen(
				path.c_str(), "w8") };
		if (tif == nullptr)
			throw std::runtime_error { "savers::TIFF: Could not open file "
					+ path + " for writing." };

		for (auto i = 0u; i < volume.depth(); ++i) {
			TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, volume.width());
			TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, volume.height());
			TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE,
					detail::BitsPerSample<value_type>::value);
			TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT,
					detail::SampleFormat<value_type>::value);

			auto slice = volume[i];
			auto data = slice.data();
			auto dataPtr = data;

			for (auto row = 0u; row < slice.height(); ++row) {
				TIFFWriteScanline(tif.get(), dataPtr, row);
				dataPtr += volume.width();
			}

			if (TIFFWriteDirectory(tif.get()) != 1)
				throw std::runtime_error {
						"savers::TIFF: tiffio error while writing to " + path };
		}
	}

protected:
	~TIFF() = default;

	double lastSeconds_;

};
}
}

#endif /* SAVERS_TIFF_H_ */
