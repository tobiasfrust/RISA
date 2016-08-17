/*
 * Copyright 2016 Tobias Frust
 *
 * CircularBuffer.h
 *
 *  Created on: 16.06.2016
 *      Author: Tobias Frust
 */

#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <vector>
#include <cstddef>

namespace ddrf {

   template <class Object>
   class CircularBuffer{
   public:
      CircularBuffer(std::size_t size) : size_{size}, count_{0u}, index_{0u}, buffer_{std::vector<Object>(size)} {}


      template <class Item>
      void push_back(Item&& item){
         if(count_ < size_)
            count_ = index_+ 1u;
         //making Image go out of scope(otherwise performance drop if buffer filled)
         auto release = std::move(buffer_[index_]);
         buffer_[index_] = std::move(item);
         index_ = (index_ + 1) % size_;
      }

      Object at(std::size_t index){
         auto ret = std::move(buffer_[index]);
         return ret;
      }

      std::size_t count(){
         return count_;
      }

      bool full(){
         return count_ >= size_;
      }

      void clear(){
         count_ = 0u;
         index_ = 0u;
      }

   private:
      const std::size_t size_;
      std::size_t count_;
      std::size_t index_;
      std::vector<Object> buffer_;
   };


}

#endif /* CIRCULARBUFFER_H_ */
