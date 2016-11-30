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
 * Authors: Tobias Frust (FWCC) <t.frust@hzdr.de>
 *
 */

#ifndef CUDA_KERNELS_FILTER_H_
#define CUDA_KERNELS_FILTER_H_

#define _USE_MATH_DEFINES
#include <cmath>

namespace risa {
namespace cuda {

__global__ void filterSL(int x, int y, cufftComplex *data) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < x && j < y) {
		const float w = (float) i / (float) (x - 1) * M_PI;
		const float divisor = 2.0 * (x - 1);
		float temp;
		if (i == 0)
			temp = 0.0;
		else
			temp = w * fabsf(sinf(w) / (M_PI * w));
		data[i + j * x].x *= temp / divisor;
		data[i + j * x].y *= temp / divisor;
	}
}

__global__ void filterRamp(int x, int y, cufftComplex *data) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < x && j < y) {
		//Rampenfilter
		const float divisor = 2 * (x - 1);
		data[i + j * x].x *= (float) i / (x - 1) * M_PI / divisor;
		data[i + j * x].y *= (float) i / (x - 1) * M_PI / divisor;
	}
}

__global__ void filterHamming(int x, int y, cufftComplex *data) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < x && j < y) {
		float temp;
		const float divisor = 2 * (x - 1);
		const float w = (float) i / (x - 1) * M_PI;
		if (i == 0)
			temp = 0.0;
		else
			temp = w * (0.54 + 0.46 * cosf(w));
		data[i + j * x].x *= temp / divisor;
		data[i + j * x].y *= temp / divisor;
	}
}

__global__ void filterHanning(int x, int y, cufftComplex *data) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < x && j < y) {
		float temp;
		const float divisor = 2 * (x - 1);
		const float w = (float) i / (x - 1) * M_PI;
		if (i == 0)
			temp = 0.0;
		else
			temp = w * (0.5 + 0.5 * cosf(w));
		data[i + j * x].x *= temp / divisor;
		data[i + j * x].y *= temp / divisor;
	}
}

//!	computes the value of the Shepp-Logan-filter function
/**
 *	@param[in]	w	the coordinate at the frequency axis
 *	@param[in]	d	the cutoff-fraction
 *
 *
 */
template <typename T>
auto inline sheppLogan(const T w, const T d) -> T {
   const T ret = std::sin(w/(2.0*d))/(w/(2.0*d));
   return ret;
}

//!	computes the value of the Cosine-filter function
/**
 *	@param[in]	w	the coordinate at the frequency axis
 *	@param[in]	d	the cutoff-fraction
 *
 *
 */
template <typename T>
auto inline cosine(const T w, const T d) -> T {
   const T ret = std::cos(w/(2.0*d));
   return ret;
}

//!	computes the value of the Hamming-filter function
/**
 *	@param[in]	w	the coordinate at the frequency axis
 *	@param[in]	d	the cutoff-fraction
 *
 *
 */
template <typename T>
auto inline hamming(const T w, const T d) -> T {
   const T ret = 0.54 + 0.46 * std::cos(w/d);
   return ret;
}

//!	computes the value of the Hanning-filter function
/**
 *	@param[in]	w	the coordinate at the frequency axis
 *	@param[in]	d	the cutoff-fraction
 *
 *
 */
template <typename T>
auto inline hanning(const T w, const T d) -> T {
   const T ret = (1 + std::cos(w/d))/ 2.;
   return ret;
}

}
}

#endif /* CUDA_KERNELS_FILTER_H */
