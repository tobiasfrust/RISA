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
