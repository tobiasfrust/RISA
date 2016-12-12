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

#ifndef GLADOS_IMAGESAVER_H_
#define GLADOS_IMAGESAVER_H_

#include <string>
#include <utility>

#include "Image.h"
#include "Volume.h"

namespace glados
{
	template <class Implementation>
	class ImageSaver : public Implementation
	{
		public:
			using manager_type = typename Implementation::manager_type;

		public:
			ImageSaver(const std::string& path) : Implementation(path) {}

			/*
			 * Saves an image to the given path.
			 */
			auto saveImage(Image<manager_type> image, std::string path) -> void
			{
				Implementation::saveImage(std::move(image), path);
			}

			/*
			 * Saves a volume to the given path.
			 */
			auto saveVolume(Volume<manager_type> volume, std::string path) -> void
			{
				Implementation::saveVolume(std::move(volume), path);
			}

		protected:
			~ImageSaver() = default;

	};
}

#endif /* GLADOS_IMAGESAVER_H_ */
