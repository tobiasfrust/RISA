/*
 * Copyright 2016
 *
 * OnlineReceiverNotification.h
 *
 *  Created on: 11.07.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
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
