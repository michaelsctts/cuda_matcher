#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void findNearestNeighbors(const float* descriptors0,
                                     const float* descriptors1, int* matches,
                                     float* scores, int nDescriptors,
                                     float ratio_thresh_sq,
                                     float distance_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors) {
    float minDistance = FLT_MAX;
    int nearestNeighborIdx = -1;

    for (int i = 0; i < nDescriptors; ++i) {
      float distance = 0.0f;
      for (int j = 0; j < 128; ++j) {
        float diff = descriptors0[idx * 128 + j] - descriptors1[i * 128 + j];
        distance += diff * diff;
      }

      if (distance < minDistance) {
        minDistance = distance;
        nearestNeighborIdx = i;
      }
    }

    float distance_thresh_sq_check = distance_thresh_sq * distance_thresh_sq;
    float ratio_thresh_sq_check = ratio_thresh_sq * ratio_thresh_sq;
    bool validMatch = (minDistance <= distance_thresh_sq_check);
    if (validMatch && ratio_thresh_sq_check > 0) {
      float secondDistance = FLT_MAX;
      for (int i = 0; i < nDescriptors; ++i) {
        if (i != nearestNeighborIdx) {
          float distance = 0.0f;
          for (int j = 0; j < 128; ++j) {
            float diff =
                descriptors0[idx * 128 + j] - descriptors1[i * 128 + j];
            distance += diff * diff;
          }
          secondDistance =
              (distance < secondDistance) ? distance : secondDistance;
        }
      }
      validMatch = (minDistance <= ratio_thresh_sq_check * secondDistance);
    }

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch)
                      ? (1.0f - minDistance / distance_thresh_sq_check) / 2.0f
                      : 0.0f;
  }
}
__global__ void mutualCheck(const int* matches0, const int* matches1,
                            int* matches, int nDescriptors) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors) {
    int match1 = matches0[idx];
    int match2 = (match1 != -1) ? matches1[match1] : -1;
    matches[idx] = (match2 == idx) ? match1 : -1;
  }
}

void featureMatching(const std::vector<float>& descriptors0,
                     const std::vector<float>& descriptors1,
                     std::vector<int>& matches, std::vector<float>& scores,
                     float ratio_thresh, float distance_thresh,
                     int nDescriptors) {
  float *d_descriptors0, *d_descriptors1;
  cudaMalloc(&d_descriptors0, descriptors0.size() * sizeof(float));
  cudaMalloc(&d_descriptors1, descriptors1.size() * sizeof(float));
  cudaMemcpy(d_descriptors0, descriptors0.data(),
             descriptors0.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_descriptors1, descriptors1.data(),
             descriptors1.size() * sizeof(float), cudaMemcpyHostToDevice);

  int *d_matches0, *d_matches1, *d_matches;
  float* d_scores;
  cudaMalloc(&d_matches0, nDescriptors * sizeof(int));
  cudaMalloc(&d_matches1, nDescriptors * sizeof(int));
  cudaMalloc(&d_matches, nDescriptors * sizeof(int));
  cudaMalloc(&d_scores, nDescriptors * sizeof(float));

  int threadsPerBlock = 32;
  int blocksPerGrid = (nDescriptors + threadsPerBlock - 1) / threadsPerBlock;
  // we calculate the matches for the first image
  findNearestNeighbors<<<blocksPerGrid, threadsPerBlock>>>(
      d_descriptors0, d_descriptors1, d_matches0, d_scores, nDescriptors,
      ratio_thresh * ratio_thresh, distance_thresh * distance_thresh);

  // cudaDeviceSynchronize();
  // we calculate the matches for the second image
  findNearestNeighbors<<<blocksPerGrid, threadsPerBlock>>>(
      d_descriptors1, d_descriptors0, d_matches1, d_scores, nDescriptors,
      ratio_thresh * ratio_thresh, distance_thresh * distance_thresh);

  cudaDeviceSynchronize();

  // we check if the matches are mutual
  mutualCheck<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                  d_matches, nDescriptors);

  cudaDeviceSynchronize();

  cudaMemcpy(matches.data(), d_matches, nDescriptors * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(scores.data(), d_scores, nDescriptors * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_descriptors0);
  cudaFree(d_descriptors1);
  cudaFree(d_matches0);

  cudaFree(d_matches1);
  cudaFree(d_matches);
  cudaFree(d_scores);
}