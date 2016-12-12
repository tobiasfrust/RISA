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

#ifndef MEMORY_H_
#define MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <memory>

#include "../Memory.h"

namespace glados
{
	namespace def
	{
		class copy_policy
		{
			protected:
				~copy_policy() = default;

				/* 1D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t size) -> void
				{
					std::copy(src.get(), src.get() + size, dest.get());
				}

				/* 2D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height) -> void
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}

				/* 3D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height, std::size_t depth) -> void
				{
					std::copy(src.get(), src.get() + width * height * depth, dest.get());
				}
		};

		template <class T> using ptr = glados::ptr<T, copy_policy, std::unique_ptr<T[]>>;
		template <class T, class is3D> using pitched_ptr = glados::pitched_ptr<T, copy_policy, is3D, std::unique_ptr<T[]>>;
	}
}



#endif /* MEMORY_H_ */
