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

#ifndef PERFORMANCE_H_
#define PERFORMANCE_H_

#include <chrono>

namespace risa {

//! The Timer-class uses the std::chrono library and provides a high precision timer.
/**
* This class provides a high precision timer based on the chrono-library of C++11.
* It can be used, to measure the elapsed time.
*/
class Timer {

public:
   Timer() {
   }

   //! Start the time duration measurement.
   void start() {
      beg_ = clock_::now();
   }

   //! Stop the time duration measurement.
   void stop() {
      end_ = clock_::now();
   }

   //! Computes the elapsed time between #start() and #stop()
   /**
    * @return  the elapsed time duration between #start() and #stop()
    */
   double elapsed() const {
      return std::chrono::duration_cast < second_ > (end_ - beg_).count();
   }

private:
   typedef std::chrono::high_resolution_clock clock_;
   typedef std::chrono::duration<double, std::ratio<1> > second_;
   std::chrono::time_point<clock_> beg_;  //!<  stores the beginning point
   std::chrono::time_point<clock_> end_;  //!<  stores the end time point
};

}

#endif /* PERFORMANCE_H_ */
