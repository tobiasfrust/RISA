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

namespace risa{

class Timer
{
public:
    Timer(){}
    void start() { beg_ = clock_::now(); }
    void stop()  { end_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (end_ - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
    std::chrono::time_point<clock_> end_;
};

}

#endif /* PERFORMANCE_H_ */
