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
#ifndef MEMORYPOOL_H_
#define MEMORYPOOL_H_

#include "Singleton.h"
#include "Image.h"

#include <boost/log/trivial.hpp>

#include <vector>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <memory>

namespace glados {

template<class MemoryManager>
class Image;

//! This class acts as a Memory pool and initializes memory at program initialization
/**
 *	At program initialization the requesting stage asks for a given number of
 *	elements of a given data type and size. The MemoryPool allocates the memory
 *	and provides during data processing, when a stage asks for it.
 *
 */
template<class MemoryManager>
class MemoryPool: public Singleton<MemoryPool<MemoryManager>>, MemoryManager {

	friend class Singleton<MemoryPool<MemoryManager>> ;
public:
	//forward declaration
	using type = glados::Image<MemoryManager>;

	//! Returns memory during data processing to the requesting stage.
	/**
	 * All stages that are registered in MemoryPool can request memory with
	 * this function. If the stage is not registered, an exception will be thrown.
	 * Memory allocation occurs only, if stage did not request enough memory
	 * during registration. In all other cases no allocation, no copy operations
	 * will be performed.
	 *
	 * @param[in] idx stage that requests memory, got an id during registration.
	 *            This id needs to passed to this function.
	 */
	auto requestMemory(unsigned int idx) -> type {
		//std::lock_guard<std::mutex> lock(memoryManagerMutex_);
      auto lock = std::unique_lock<std::mutex>{memoryManagerMutex_};
		if(memoryPool_.size() <= idx)
		    throw std::runtime_error("cuda::MemoryPool: Stage needs to be registered first.");
		while(memoryPool_[idx].empty()){
		   cv_.wait(lock);
		}
		auto ret = std::move(memoryPool_[idx].back());
		memoryPool_[idx].pop_back();
		return ret;
	}

	//!	This function reenters the data element in the memory pool.
	/**
	 * This function gets an image, e.g. when image gets out of scope
	 * and stores it in the memory pool vector, where it originally
	 * came from
	 *
	 * @param[in] img Image, that shall be returned into memory pool for reuse
	 *
	 */
	auto returnMemory(type&& img) -> void {
		if(memoryPool_.size() <= img.memoryPoolIndex())
		   throw std::runtime_error("cuda::MemoryPool: Stage needs to be registered first.");
      std::lock_guard<std::mutex> lock{memoryManagerMutex_};
		memoryPool_[img.memoryPoolIndex()].push_back(std::move(img));
		cv_.notify_one();
	}

	//! This function is called at program initialization, when a stage needs memory during data processing.
	/**
	 * All stages that need memory need to register in MemoryManager.
	 * Stages need to tell, which size of memory they need and how many elements.
	 * The MemoryManager then allocates the memory and manages it.
	 *
	 * @param[in] numberOfElements 	number of elements that shall be allocated by the MemoryManager
	 * @param[in] size				size of memory that needs to be allocated per element
	 *
	 * @return identifier, where
	 *
	 */
	auto registerStage(const int& numberOfElements,
			const size_t& size) -> int {
		//lock, to ensure thread safety
	   std::lock_guard<std::mutex> lock(memoryManagerMutex_);
		std::vector<type> memory;
		int index = memoryPool_.size();
		for(int i = 0; i < numberOfElements; i++) {
			auto img = type {};
			auto ptr = MemoryManager::make_ptr(size);
			img = type {size, 0, 0, std::move(ptr)};
			img.setMemPoolIdx(index);
			memory.push_back(std::move(img));
		}
		memoryPool_.push_back(std::move(memory));
		return index;
	}

	//! When the classes are destroyed, this functions frees the allocated memory.
	/**
	 *	@param[in]	idx	idx stage that requests memory, got an id during registration.
	 *            			This id needs to passed to this function.
	 *
	 */
	auto freeMemory(const unsigned int idx) -> void {
	   for(auto& ele: memoryPool_[idx]){
	      ele.invalid();
	   }
	   memoryPool_[idx].clear();
	}

private:
	~MemoryPool() = default;

	MemoryPool() = default;

private:
	std::vector<std::vector<type>> memoryPool_;	//!	this vector stores the elements for each stage
	mutable std::mutex memoryManagerMutex_;		//! 	the mutex to ensure thread-safety
	std::condition_variable cv_;						//!	condition_variable to notify threads
};


}

#endif /* MEMORYPOOL_H_ */
