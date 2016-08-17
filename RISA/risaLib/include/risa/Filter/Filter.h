/*
 * Copyright 2016
 *
 * Filter.h
 *
 *  Created on: 21.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thrust/device_vector.h>

#include <cufft.h>

#include <map>
#include <thread>

namespace risa {
namespace cuda {

   namespace detail{
      enum FilterType: short {
         ramp,
         sheppLogan,
         cosine,
         hamming,
         hanning
      };
   }

class Filter {
public:
	using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
	using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;

public:
	Filter(const std::string& configFile);

	~Filter();

	auto process(input_type&& sinogram) -> void;
	auto wait() -> output_type;

protected:

private:
	std::map<int, ddrf::Queue<input_type>> sinograms_;
	ddrf::Queue<output_type> results_;

	std::map<int, std::thread> processorThreads_;

	int numberOfProjections_;
	int numberOfDetectors_;
	int numberOfPixels_;

	detail::FilterType filterType_;
	float cutoffFraction_;

	//on which device shall it be executed(used for multi-staged pipeline)
	int numberOfDevices_;

   //kernel execution coniguration
   int blockSize2D_;

	std::map<int, cufftHandle> plansFwd_;
	std::map<int, cufftHandle> plansInv_;

	std::map<int, cudaStream_t> streams_;

	std::vector<float> filter_;

	auto processor(const int deviceID) -> void;
	auto initCuFFT(const int deviceID) -> void;
	auto readConfig(const std::string& configFile) -> bool;
	auto designFilter() -> void;
};
}
}

#endif /* FILTER_H_ */
