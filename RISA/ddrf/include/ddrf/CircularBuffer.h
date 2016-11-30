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
 *
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
