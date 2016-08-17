/*
 * Copyright 2016
 *
 * D2H.h
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef D2H_H_
#define D2H_H_

#include "../Basics/performance.h"

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include <thread>
#include <map>

namespace risa {
namespace cuda {

class D2H {
public:
	using hostManagerType = ddrf::cuda::HostMemoryManager<float, ddrf::cuda::async_copy_policy>;
	using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float, ddrf::cuda::async_copy_policy>>;
	using output_type = ddrf::Image<ddrf::cuda::HostMemoryManager<float, ddrf::cuda::async_copy_policy>>;

public:
	D2H(const std::string& configFile);
	~D2H();

	auto process(input_type&& img) -> void;
	auto wait() -> output_type;

private:
	std::map<int, ddrf::Queue<input_type>> imgs_;
	ddrf::Queue<output_type> results_;

	std::map<int, std::thread> processorThreads_;
  std::map<int, cudaStream_t> streams_;

	//on which device shall it be executed(used for multi-staged pipeline)
	unsigned int memoryPoolIdx_;

	unsigned int averageCounter_;

	int memPoolSize_;

	int numberOfDevices_;
	int numberOfPixels_;

	std::size_t count_{0};

	double reconstructionRate_;
	double counter_;

	Timer tmr_;

	auto processor(const int deviceID) -> void;
	auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* D2H_H_ */
