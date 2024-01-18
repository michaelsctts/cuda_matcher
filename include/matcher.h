#pragma once

#include "datatypes.h"

// NN-mutual matching
void featureMatching(const std::vector<float> &descriptors0,
                     const std::vector<float> &descriptors1,
                     std::vector<int> &matches, std::vector<float> &scores,
                     float ratio_thresh_sq, int nDescriptors0,
                     int nDescriptors1);

void featureMatchingLegacy(const std::vector<float> &descriptors0,
                           const std::vector<float> &descriptors1,
                           std::vector<int> &matches,
                           std::vector<float> &scores, float ratio_thresh_sq,
                           int nDescriptors0, int nDescriptors1);
