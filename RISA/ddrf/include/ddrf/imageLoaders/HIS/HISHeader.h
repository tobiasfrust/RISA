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

/*
 * HISHeader.h
 *
 *  Created on: 26.11.2015
 *      Author: Jan Stephan
 *
 *      Header definition for the HIS file format.
 *
 *      Types derived from old implementation:
 *			BYTE -> std::uint8_t
 *			DWORD -> std::uint32_t
 *      	WORD -> std::uint16_t
 *      	ULONG -> std::uint32_t
 */

#ifndef HISHEADER_H_
#define HISHEADER_H_

#include <cstdint>

namespace ddrf
{
	namespace loaders
	{
		enum class HISConst : std::uint16_t
		{
			file_header_size = 68,
			rest_size = 34,
			hardware_header_size = 32,
			header_size = file_header_size + hardware_header_size,
			file_id = 0x7000
		};

		struct HISHeader
		{
			std::uint16_t file_type;											// == HISConst::file_id
			std::uint16_t header_size;											// size of this file header in bytes
			std::uint16_t header_version;										// yy.y
			std::uint32_t file_size;											// size of the whole file in bytes
			std::uint16_t image_header_size;									// size of the image header in bytes
			std::uint16_t ulx, uly, brx, bry;									// bounding rectangle of image
			std::uint16_t number_of_frames;
			std::uint16_t correction;											// 0 = none, 1 = offset, 2 = gain, 4 = bad pixel, (ored)
			double integration_time;											// frame time in microseconds
			std::uint16_t type_of_numbers;										// short, long integer, float, signed/unsigned, inverted
																				// fault map, offset/gain correction data,
																				// badpixel correction data
			std::uint8_t x[static_cast<std::uint32_t>(HISConst::rest_size)];	// fill up to 68 byte
		};
	}
}


#endif /* HISHEADER_H_ */
