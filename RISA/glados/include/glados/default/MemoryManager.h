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

#ifndef DEF_MEMORYMANAGER_H_
#define DEF_MEMORYMANAGER_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "Memory.h"

namespace glados
{
	namespace def
	{
		template <class T>
		class MemoryManager
		{
			public:
				using value_type = T;
				using size_type = std::size_t;
				using pointer_type_1D = ptr<T>;
				using pointer_type_2D = pitched_ptr<T, std::false_type>;
				using pointer_type_3D = pitched_ptr<T, std::true_type>;

			public:
				inline auto make_ptr(size_type size) -> pointer_type_1D
				{
					auto ptr = std::unique_ptr<T[]>(new value_type[size]);
					return ptr<T, std::false_type>(std::move(ptr), size * sizeof(T));
				}

				inline auto make_ptr(size_type width, size_type height) -> pointer_type_2D
				{
					auto ptr = std::unique_ptr<T[]>(new value_type[width * height]);
					return pitched_ptr<T, std::false_type>(std::move(ptr), width * sizeof(T), width, height);
				}

				inline auto make_ptr(size_type width, size_type height, size_type depth) -> pointer_type_3D
				{
					auto ptr = std::unique_ptr<T>(new value_type[width * height * depth]);
					return pitched_ptr<T, std::true_type>(std::move(ptr), width * sizeof(T), width, height, depth);
				}

				inline auto copy(pointer_type_1D& dest, const pointer_type_1D& src, size_type size) -> void
				{
					std::copy(src.get(), src.get() + size, dest.get());
				}

				inline auto copy(pointer_type_2D& dest, const pointer_type_2D& src, size_type width, size_type height) -> void
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}

				inline auto copy(pointer_type_3D& dest, const pointer_type_3D& src, size_type width, size_type height, size_type depth) -> void
				{
					std::copy(src.get(), src.get() + width * height * depth, dest.get());
				}
		};
	}
}

#endif /* DEF_MEMORYMANAGER_H_ */
