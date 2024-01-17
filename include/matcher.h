#pragma once

#include "image.h"

// NN-mutual matching
void featureMatching(const std::vector<float> &descriptors0,
                     const std::vector<float> &descriptors1,
                     std::vector<int> &matches, std::vector<float> &scores,
                     float ratio_thresh, float distance_thresh,
                     int nDescriptors);
