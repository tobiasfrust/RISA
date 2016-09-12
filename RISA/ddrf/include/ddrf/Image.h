/*
 * Image.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      Image class that holds a pointer to the concrete image data
 *      The individual objects are usually created by the ImageHandler and its corresponding handler policies.
 */

#ifndef DDRF_IMAGE_H_
#define DDRF_IMAGE_H_

#include "MemoryPool.h"

#include <boost/log/trivial.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <time.h>

namespace ddrf {
template<class MemoryManager>
class Image: public MemoryManager {
public:
   using value_type = typename MemoryManager::value_type;
   using pointer_type = typename MemoryManager::pointer_type_1D;
   using size_type = typename MemoryManager::size_type;
private:
   typedef std::chrono::high_resolution_clock clock_;

public:
   Image() noexcept
   : size_ {0}, index_ {0}, memoryPoolIndex_(0), plane_ {0}, data_ {nullptr}, valid_ {false}
   {
   }

   ~Image() {
      if(valid_) {
         //if valid image goes out of scope, return it to MemoryPool -> automatic reuse
         MemoryPool<MemoryManager>::instance()->returnMemory(std::move(*this));
      }
   }

   Image(size_type size, size_type idx = 0, size_type planeID = 0,
         pointer_type img_data = nullptr)
   : MemoryManager()
   , size_ {size}, index_ {idx}, memoryPoolIndex_(0), plane_(planeID), data_ {std::move(img_data)}, valid_ {true}
   {
      if(data_ == nullptr)
      data_ = MemoryManager::make_ptr(size_);
   }

   Image(const Image& other)
   : MemoryManager(other)
   , size_ {other.size_}, index_ {other.index_}, plane_ {other.plane_}, memoryPoolIndex_(other.memoryPoolIndex_), valid_ {other.valid_}
   {
      if(other.data_ == nullptr)
      data_ = nullptr;
      else
      {
         data_ = MemoryManager::make_ptr(size_);
         MemoryManager::copy(data_, other.data_, size_);
      }
   }

   auto setIdx(unsigned int idx) -> void {
      index_ = idx;
   }

   auto setPlane(size_type plane) -> void {
      plane_ = plane;
   }

   auto setMemPoolIdx(unsigned int idx) -> void {
      memoryPoolIndex_ = idx;
   }

   auto setStart(std::chrono::time_point<clock_> start) -> void {
      start_ = start;
   }

   auto duration() -> double {
      return std::chrono::duration<double,std::milli>(clock_::now() - start_).count();
   }

   auto invalid() -> void {
      valid_ = false;
   }

   template <typename U>
   auto operator=(const Image<U>& rhs) -> Image&
   {
      size_ = rhs.size();
      index_ = rhs.index();
      valid_ = rhs.valid();
      plane_ = rhs.plane();
      start_ = rhs.start();

      if(rhs.container() == nullptr)
      data_ = nullptr;
      else
      {
         data_.reset(); // delete old content if any
         data_ = MemoryManager::make_ptr(size_);
         MemoryManager::copy(data_, rhs.container(), size_);
      }

      return *this;
   }

   Image(Image&& other) noexcept
   : MemoryManager(std::move(other))
   , size_ {other.size_}, index_ {other.index_}, data_ {std::move(other.data_)}
   , valid_ {other.valid_}, plane_{other.plane_}, memoryPoolIndex_ {other.memoryPoolIndex_}, start_(other.start_)
   {
      other.valid_ = false; // invalid after we moved its data
   }

   auto operator=(Image&& rhs) noexcept -> Image&
   {
      size_ = rhs.size_;
      index_ = rhs.index_;
      data_ = std::move(rhs.data_);
      plane_ = rhs.plane_;
      valid_ = rhs.valid_;
      start_ = rhs.start_;
      memoryPoolIndex_ = rhs.memoryPoolIndex_;

      MemoryManager::operator=(std::move(rhs));

      rhs.valid_ = false;
      return *this;
   }

   auto size() const noexcept -> size_type
   {
      return size_;
   }

   /*
    * returns a non-owning pointer to the data. Do not delete this pointer as the Image object will take
    * care of the memory.
    */
   auto data() const noexcept -> value_type*
   {
      return data_.get();
   }

   auto pitch() const noexcept -> size_type
   {
      return data_.pitch();
   }

   auto valid() const noexcept -> bool
   {
      return valid_;
   }

   auto index() const noexcept -> size_type
   {
      return index_;
   }

   auto plane() const noexcept -> size_type {
      return plane_;
   }

   auto memoryPoolIndex() const noexcept -> size_type {
      return memoryPoolIndex_;
   }

   auto start() const noexcept -> std::chrono::time_point<clock_> {
      return start_;
   }

   /*
    * return the underlying pointer
    */
   auto container() const noexcept -> const pointer_type&
   {
      return data_;
   }

private:
   size_type size_;
   size_type index_;
   size_type plane_;
   pointer_type data_;
   size_type memoryPoolIndex_;
   std::chrono::time_point<clock_> start_;
   bool valid_;
};
}

#endif /* DDRF_IMAGE_H_ */
