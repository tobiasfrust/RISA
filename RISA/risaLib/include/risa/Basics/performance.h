/*
 * Copyright 2016
 *
 * performance.h
 *
 *  Created on: 19.04.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 *  http://stackoverflow.com/questions/2808398/easily-measure-elapsed-time
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
   std::chrono::time_point<clock_> beg_;
   std::chrono::time_point<clock_> end_;
};

}

#endif /* PERFORMANCE_H_ */
