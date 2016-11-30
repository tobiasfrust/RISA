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
 */

#ifndef CUDA_COORDINATES_H_
#define CUDA_COORDINATES_H_

namespace ddrf
{
	namespace cuda
	{
		inline __device__ auto getX() -> unsigned int
		{
			return blockIdx.x * blockDim.x + threadIdx.x;
		}

		inline __device__ auto getY() -> unsigned int
		{
			return blockIdx.y * blockDim.y + threadIdx.y;
		}

		inline __device__ auto getZ() -> unsigned int
		{
			return blockIdx.z * blockDim.z + threadIdx.z;
		}
	}
}




#endif /* COORDINATES_H_ */
