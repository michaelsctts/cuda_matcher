#pragma once

#include "datatypes.h"

void featureMatching(const float* d_descriptors0, const float* d_descriptors1,
                     std::vector<int>& matches, std::vector<float>& scores,
                     float ratio_thresh_sq, int nDescriptors0,
                     int nDescriptors1);

// TODO: implement this
void featureMatchingHalf(float* d_descriptors0, float* d_descriptors1,
                         std::vector<int>& matches, std::vector<float>& scores,
                         float ratio_thresh_sq, int nDescriptors0,
                         int nDescriptors1);

void allocateDescriptors(float** d_descriptors,
                         const std::vector<float>& descriptors);

void deallocateDescriptors(float* d_descriptors);
