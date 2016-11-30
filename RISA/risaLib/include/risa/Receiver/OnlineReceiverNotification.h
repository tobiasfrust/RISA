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

#ifndef ONLINERECEIVERNOTIFICATION_H_
#define ONLINERECEIVERNOTIFICATION_H_

#include <boost/log/trivial.hpp>

#include <vector>
#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace risa{

//! This class implements the synchronization between the ReceiverModule class and the superior Receiver class
   class OnlineReceiverNotification {

   public:
      OnlineReceiverNotification(int size) : size_{size}, lastIndex_(0u)
      {
         BOOST_LOG_TRIVIAL(debug) << "Resizing to " << size << " elements.";
         indeces_.resize(size);
      }

      //! checks, if complete new sinogram is available
      /**
       * Therefore, this function finds the minimum element in @indices_ and returns it.
       * For the minimum element, it can be guaranteed, that the sinogram is complete
       *
       * @return the index of the last complete sinogram available in the buffers
       */
      std::size_t fetch(){
         BOOST_LOG_TRIVIAL(debug) << "Indices size: " << indeces_.size();
         auto lock = std::unique_lock<std::mutex>{mutex_};
         auto minElement = std::min_element(std::begin(indeces_), std::end(indeces_));
         while(minElement == std::end(indeces_) || *minElement <= lastIndex_){
            cv_.wait(lock);
            minElement = std::min_element(std::begin(indeces_), std::end(indeces_));
         }
         BOOST_LOG_TRIVIAL(debug) << "########### SINO " << *minElement << " complete.";
         lastIndex_ = *minElement;
         return *minElement;
      }

      //! This function inserts the  new sinogram index in #indeces_ and notifies the Receiver about the arrival
      /**
       * @param[in]  receiverID  the id of the ReceiverModule, that received the sinogram
       * @param[in]  index       the index of the sinogram, that has just arrived
       */
      void notify(int receiverID, std::size_t index){
         //lock mutex
         std::lock_guard<std::mutex> lock(mutex_);
         //only insert new index if index is greater (wrong package order)
         if(indeces_[receiverID] < index)
            indeces_[receiverID] = index;
         //notify collector thread to check if new sinogram can be fetched
         cv_.notify_one();
      }

   private:

      std::vector<std::size_t> indeces_;
      int size_;
      std::mutex mutex_;
      std::condition_variable cv_;
      std::size_t lastIndex_;

   };

}

#endif /* ONLINERECEIVERNOTIFICATION_H_ */
