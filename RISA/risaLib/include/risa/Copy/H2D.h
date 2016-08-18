/*
 * Copyright 2016
 *
 * H2D.h
 *
 *  Created on: 28.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 */

#ifndef H2D_H_
#define H2D_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Memory.h>

#include "../Basics/performance.h"

#include <thread>
#include <vector>
#include <map>
#include <mutex>

namespace risa {
namespace cuda {
	
class H2D {
public:
	using input_type = ddrf::Image<ddrf::cuda::HostMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
	using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>>;
	using deviceManagerType = ddrf::cuda::DeviceMemoryManager<unsigned short, ddrf::cuda::async_copy_policy>;

public:
	H2D(const std::string& configFile);
	~H2D();

	auto process(input_type&& sinogram) -> void;
	auto wait() -> output_type;

private:

	std::map<int, ddrf::Queue<input_type>> sinograms_;
	ddrf::Queue<output_type> results_;

	std::map<int, std::thread> processorThreads_;
	std::map<int, cudaStream_t> streams_;
	std::map<int, unsigned int> memoryPoolIdxs_;

	mutable std::mutex mutex_;

	auto processor(int deviceID) -> void;

	double worstCaseTime_;
	double bestCaseTime_;
	Timer tmr_;

	std::size_t lastIndex_;
	std::size_t lostSinos_;

	std::size_t count_{0};

	int lastDevice_;

	int numberOfDevices_;

	int numberOfDetectors_, numberOfProjections_;

	int memPoolSize_;

	auto readConfig(const std::string& configFile) -> bool;
};
}
}

#endif /* D2H_H_ */
