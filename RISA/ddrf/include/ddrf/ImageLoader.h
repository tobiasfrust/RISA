#ifndef DDRF_IMAGELOADER_H_
#define DDRF_IMAGELOADER_H_

#include <cstddef>
#include <memory>
#include <string>

#include "Image.h"

namespace ddrf
{
	template <class Implementation>
	class ImageLoader : public Implementation
	{
		public:
			using manager_type = typename Implementation::manager_type;

		public:

			ImageLoader(const std::string& address, const std::string& configPath) : Implementation(address, configPath){}

			/*
			 * Loads an image from the given path.
			 * */
			auto loadImage() -> Image<manager_type>
			{
				return Implementation::loadImage();
			}
	};
}



#endif /* DDRF_IMAGELOADER_H_ */
