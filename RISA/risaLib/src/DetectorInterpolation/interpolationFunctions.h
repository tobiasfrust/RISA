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
 * Authors: Tobias Frust (FWCC) <t.frust@hzdr.de>
 *
 */
#ifndef INTERPOLATIONFUNCTIONS_H_
#define INTERPOLATIONFUNCTIONS_H_

#include <vector>

template <typename T>
auto findDefectDetectors(T* data, std::vector<double>& filterFunction, std::vector<int>& defectDetectors,
      unsigned int numberOfDetectors, unsigned int numberOfProjections, double threshMin,
      double threshMax) -> void {

   std::vector<double> var(numberOfDetectors, 0.0);
   unsigned int scale = 2;

   for(auto detInd = 0u; detInd < numberOfDetectors; detInd++){
      double varMax = data[detInd];
      double varMin = varMax;
      for(auto projInd = 0u; projInd < numberOfProjections - 1; projInd++){
        var[detInd] += std::abs((double)data[detInd + projInd*numberOfDetectors] - (double)data[detInd + (projInd+1)*numberOfDetectors]);
        if(data[detInd + projInd*numberOfDetectors] > varMax)
           varMax = (double)data[detInd + projInd*numberOfDetectors];
        if(data[detInd + projInd*numberOfDetectors] < varMin)
           varMin = (double)data[detInd + projInd*numberOfDetectors];
      }
      var[detInd] *= std::pow(varMax - varMin, scale);
   }
   int addNeighboursToFlickering{2};
   for(auto detectorSeg = 0; detectorSeg < 2; detectorSeg++){
      for(auto i = 0u; i < numberOfDetectors/2; i++){
         double thresh_segment = 0.0;
         for(auto j = 0; j < 9; j++){
            int ind = (i - j) % (numberOfDetectors/2);
            thresh_segment += filterFunction[j] * var[ind + detectorSeg * (numberOfDetectors/2)];
            ind = (i + j) % (numberOfDetectors/2);
            thresh_segment += filterFunction[j] * var[ind + detectorSeg * (numberOfDetectors/2)];
         }
         unsigned int detInd = detectorSeg * (numberOfDetectors/2) + i;
         if(var[detInd] < threshMin*thresh_segment){
            defectDetectors[detInd] =  1;
            //std::cout << "Thresh: " << threshMin_*thresh_segment << "Defect: " << detInd << std::endl;
         }
         if(var[detInd] > threshMax * thresh_segment){
            for(int offset = -addNeighboursToFlickering; offset <= addNeighboursToFlickering; offset++){
               defectDetectors[(detInd+offset)%numberOfDetectors] = 1;
               //std::cout << "Defect: " << (detInd+offset)%numberOfDetectors_ << std::endl;
            }
         }
      }
   }
}

template <typename T>
auto interpolateDefectDetectors(T* data, std::vector<int>& defectDetectors,
      unsigned int numberOfDetectors, unsigned int numberOfProjections) -> void {
   unsigned int detID = 0;
   //interpolate
   while(detID < numberOfDetectors){
      if(defectDetectors[detID]){
         unsigned int det0 = detID;
         //BOOST_LOG_TRIVIAL(info) << "Interpolation from " << det0;
         while(defectDetectors[detID] && defectDetectors[(detID+1)%numberOfDetectors]){
            detID++;
         }
         //BOOST_LOG_TRIVIAL(info) << "to " << detID;
         for(auto i = det0; i <= detID; i++){
            float w1 = ((float)i - (float)det0 + 1.0)/((float)detID - (float)det0 + 2.0);
            float w0 = 1.0 - w1;
            for(auto projId = 0u; projId < numberOfProjections; projId++){
               data[(i%numberOfDetectors) + projId*numberOfDetectors] = w0 * data[(det0-1)%numberOfDetectors + projId*numberOfDetectors] +
                              w1 * data[(detID+1)%numberOfDetectors + projId*numberOfDetectors];
            }
         }
      }
      detID++;
   }
}


#endif /* INTERPOLATIONFUNCTIONS_H_ */
