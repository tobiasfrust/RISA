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

#ifndef CUDA_HOSTMEMORYMANAGER_H_
#define CUDA_HOSTMEMORYMANAGER_H_

#include <cstddef>
#include <type_traits>

#include "Memory.h"

namespace glados
{
	namespace cuda
	{
		template <class T, class CopyPolicy = sync_copy_policy>
		class HostMemoryManager : public CopyPolicy
		{
			public:
				using value_type = T;
				using pointer_type_1D = host_ptr<T, CopyPolicy>;
				using pointer_type_2D = pitched_host_ptr<T, CopyPolicy, std::false_type>;
				using pointer_type_3D = pitched_host_ptr<T, CopyPolicy, std::true_type>;
				using size_type = std::size_t;

			protected:
				inline auto make_ptr(size_type size) -> pointer_type_1D
				{
					return make_host_ptr<value_type, CopyPolicy>(size);
				}

				inline auto make_ptr(size_type width, size_type height) -> pointer_type_2D
				{
					return make_host_ptr<value_type, CopyPolicy>(width, height);
				}

				inline auto make_ptr(size_type width, size_type height, size_type depth) -> pointer_type_3D
				{
					return make_host_ptr<value_type, CopyPolicy>(width, height, depth);
				}

				template <typename Source>
				inline auto copy(pointer_type_1D& dest, Source& src, size_type size) -> void
				{
					CopyPolicy::copy(dest, src, size);
				}

				template <typename Source>
				inline auto copy(pointer_type_2D& dest, Source& src, size_type width, size_type height) -> void
				{
					CopyPolicy::copy(dest, src, width, height);
				}

				template <typename Source>
				inline auto copy(pointer_type_3D& dest, Source& src, size_type width, size_type height, size_type depth) -> void
				{
					CopyPolicy::copy(dest, src, width, height, depth);
				}

				~HostMemoryManager() = default;
		};
	}
}




#endif /* CUDA_HOSTMEMORYMANAGER_H_ */
