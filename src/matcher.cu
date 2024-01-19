
#include <cooperative_groups.h>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 32

__global__ void similarityMatrixAndTranspose(
    const float* descriptors0, const float* descriptors1, int nDescriptors0,
    const int nDescriptors1, const int descriptorDim, float* sim, float* simT) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    float dotProduct = 0.0f;
    for (int i = 0; i < descriptorDim; ++i) {
      dotProduct += descriptors0[idx * descriptorDim + i] *
                    descriptors1[idy * descriptorDim + i];
    }

    sim[idx * nDescriptors1 + idy] = dotProduct;
    simT[idy * nDescriptors0 + idx] = dotProduct;
  }
}

__global__ void similarityMatrixAndTransposeV2(
    const float* descriptors0, const float* descriptors1, int nDescriptors0,
    int nDescriptors1, int descriptorDim, float* sim, float* simT) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    int globalIdx = idx * descriptorDim;
    int globalIdy = idy * descriptorDim;

    float dotProduct = 0.0f;
    for (int i = 0; i < descriptorDim; ++i) {
      dotProduct += descriptors0[globalIdx + i] * descriptors1[globalIdy + i];
    }

    // Store the result in both sim and its transpose simT
    sim[idx * nDescriptors1 + idy] = dotProduct;
    simT[idy * nDescriptors0 + idx] = dotProduct;
  }
}

__global__ void find_nn(const float* sim, int* matches, float* scores,
                        const int nDescriptors0, const int nDescriptors1,
                        const float ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    float sim_nn0 = -1e30f;
    float sim_nn1 = -1e30f;
    int nearestNeighborIdx = -1;
    for (int i = 0; i < nDescriptors1; ++i) {
      if (sim[idx * nDescriptors1 + i] > sim_nn0) {
        sim_nn1 = sim_nn0;
        sim_nn0 = sim[idx * nDescriptors1 + i];
        nearestNeighborIdx = i;
      } else if (sim[idx * nDescriptors1 + i] > sim_nn1) {
        sim_nn1 = sim[idx * nDescriptors1 + i];
      }
    }

    float dist_nn0 = 2 * (1 - sim_nn0);
    float dist_nn1 = 2 * (1 - sim_nn1);

    bool validMatch = (dist_nn0 <= ratio_thresh_sq * dist_nn1);

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? (sim_nn0 + 1) / 2.0f : 0.0f;
  }
}

__global__ void find_nnV2(const float* sim, int* matches, float* scores,
                          const int nDescriptors0, const int nDescriptors1,
                          const float ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    // Load sim data into shared memory
    extern __shared__ float shared_sim[];
    for (int i = threadIdx.x; i < nDescriptors0 * nDescriptors1;
         i += blockDim.x) {
      shared_sim[i] = sim[i];
    }
    __syncthreads();

    float sim_nn0 = -1e30f;
    float sim_nn1 = -1e30f;
    int nearestNeighborIdx = -1;

    // Find nearest neighbors using parallel reduction
    for (int i = 0; i < nDescriptors1; ++i) {
      float sim_value = shared_sim[idx * nDescriptors1 + i];

      // Parallel reduction for maximum similarity value and its index
      if (sim_value > sim_nn0) {
        sim_nn1 = sim_nn0;
        sim_nn0 = sim_value;
        nearestNeighborIdx = i;
      } else if (sim_value > sim_nn1) {
        sim_nn1 = sim_value;
      }
    }

    float dist_nn0 = 2 * (1 - sim_nn0);
    float dist_nn1 = 2 * (1 - sim_nn1);

    bool validMatch = (dist_nn0 <= ratio_thresh_sq * dist_nn1);

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? ((sim_nn0 + 1) / 2.0f) : 0.0f;
  }
}

__global__ void findNearestNeighbors(const float* descriptors0,
                                     const float* descriptors1, int* matches,
                                     float* scores, int nDescriptors0,
                                     int nDescriptors1,
                                     const float ratio_thresh_sq,
                                     float distance_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    float sim0 = -100000.0f;
    int nearestNeighborIdx = -1;

    for (int i = 0; i < nDescriptors1; ++i) {
      float distance = 0.0f;
      for (int j = 0; j < 128; ++j) {
        float diff = descriptors0[idx * 128 + j] * descriptors1[i * 128 + j];
        distance += diff * diff;
      }
      if (distance > sim0) {
        sim0 = distance;
        nearestNeighborIdx = i;
      }
    }

    float sim1 = -100000.0f;
    int nearestNeighborIdx1 = -1;

    for (int i = 0; i < nDescriptors1; ++i) {
      float distance = 0.0f;
      for (int j = 0; j < 128; ++j) {
        if (i == nearestNeighborIdx) {
          continue;
        }
        float diff = descriptors0[idx * 128 + j] * descriptors1[i * 128 + j];
        distance += diff;
      }
      if (distance > sim1) {
        sim1 = distance;
        nearestNeighborIdx1 = i;
      }
    }

    // float dist_nn0 = (2 * (1 - sim0));
    // float dist_nn1 = 2 * (1 - sim1);

    // bool validMatch = true;
    // if (ratio_thresh_sq > 0) {
    bool validMatch = ((2 * (1 - sim0)) <= ratio_thresh_sq * (2 * (1 - sim1)));
    // }

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? (sim0 + 1) / 2.0f : 0.0f;
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
                     float ratio_thresh_sq, int nDescriptors0,
                     int nDescriptors1) {
  float *d_descriptors0, *d_descriptors1;
  int *d_matches0, *d_matches1, *d_matches;
  float *d_scores, *d_scores1;
  float* d_sim;
  float* d_simT;

  cudaMalloc(&d_descriptors0, descriptors0.size() * sizeof(float));
  cudaMalloc(&d_descriptors1, descriptors1.size() * sizeof(float));
  cudaMemcpy(d_descriptors0, descriptors0.data(),
             descriptors0.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_descriptors1, descriptors1.data(),
             descriptors1.size() * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&d_matches0, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_matches1, nDescriptors1 * sizeof(int));
  cudaMalloc(&d_matches, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_scores, nDescriptors0 * sizeof(float));
  cudaMalloc(&d_scores1, nDescriptors1 * sizeof(float));
  // malloc for similarity matrix
  cudaMalloc(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float));
  cudaMalloc(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float));

  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  // we calculate the matches for the first image

  // similarity matrix will be nDescriptors0 x nDescriptors1
  dim3 threadsPerBlock2D(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid2D(
      (nDescriptors0 + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
      (nDescriptors1 + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
  similarityMatrixAndTransposeV2<<<blocksPerGrid2D, threadsPerBlock2D>>>(
      d_descriptors0, d_descriptors1, nDescriptors0, nDescriptors1, 128, d_sim,
      d_simT);

  cudaDeviceSynchronize();

  // matches
  find_nn<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores,
                                              nDescriptors0, nDescriptors1,
                                              ratio_thresh_sq);

  // transpose
  // otherside matches
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  find_nn<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                               nDescriptors1, nDescriptors0,
                                               ratio_thresh_sq);

  cudaDeviceSynchronize();

  // we check if the matches are mutual
  mutualCheck<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                  d_matches, nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpy(matches.data(), d_matches, nDescriptors0 * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(scores.data(), d_scores, nDescriptors0 * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_descriptors0);
  cudaFree(d_descriptors1);
  cudaFree(d_matches0);

  cudaFree(d_sim);
  cudaFree(d_simT);

  cudaFree(d_matches1);
  cudaFree(d_matches);
  cudaFree(d_scores);
  cudaFree(d_scores1);
}

void featureMatchingLegacy(const std::vector<float>& descriptors0,
                           const std::vector<float>& descriptors1,
                           std::vector<int>& matches,
                           std::vector<float>& scores, float ratio_thresh_sq,
                           int nDescriptors0, int nDescriptors1) {
  float *d_descriptors0, *d_descriptors1;
  cudaMalloc(&d_descriptors0, descriptors0.size() * sizeof(float));
  cudaMalloc(&d_descriptors1, descriptors1.size() * sizeof(float));
  cudaMemcpy(d_descriptors0, descriptors0.data(),
             descriptors0.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_descriptors1, descriptors1.data(),
             descriptors1.size() * sizeof(float), cudaMemcpyHostToDevice);

  int *d_matches0, *d_matches1, *d_matches;
  float *d_scores, *d_scores1;

  cudaMalloc(&d_matches0, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_matches1, nDescriptors1 * sizeof(int));
  cudaMalloc(&d_matches, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_scores, nDescriptors0 * sizeof(float));
  cudaMalloc(&d_scores1, nDescriptors1 * sizeof(float));

  int threadsPerBlock = 32;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  // we calculate the matches for the first image
  findNearestNeighbors<<<blocksPerGrid, threadsPerBlock>>>(
      d_descriptors0, d_descriptors1, d_matches0, d_scores, nDescriptors0,
      nDescriptors1, ratio_thresh_sq, 0.0f);

  // cudaDeviceSynchronize();
  // we calculate the matches for the second image
  int blocksPerGrid2 = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  findNearestNeighbors<<<blocksPerGrid2, threadsPerBlock>>>(
      d_descriptors1, d_descriptors0, d_matches1, d_scores1, nDescriptors1,
      nDescriptors0, ratio_thresh_sq, 0.0f);

  cudaDeviceSynchronize();

  // we check if the matches are mutual
  mutualCheck<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                  d_matches, nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpy(matches.data(), d_matches, nDescriptors0 * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(scores.data(), d_scores, nDescriptors0 * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_descriptors0);
  cudaFree(d_descriptors1);
  cudaFree(d_matches0);

  cudaFree(d_matches1);
  cudaFree(d_matches);
  cudaFree(d_scores);
}

void allocateDescriptors(float** d_descriptors,
                         const std::vector<float>& descriptors) {
  cudaMalloc(d_descriptors, descriptors.size() * sizeof(float));
  cudaMemcpy(*d_descriptors, descriptors.data(),
             descriptors.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void featureMatchingV2(const float* d_descriptors0, const float* d_descriptors1,
                       std::vector<int>& matches, std::vector<float>& scores,
                       float ratio_thresh_sq, int nDescriptors0,
                       int nDescriptors1) {
  int *d_matches0, *d_matches1, *d_matches;
  float *d_scores, *d_scores1;
  float* d_sim;
  float* d_simT;

  cudaMalloc(&d_matches0, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_matches1, nDescriptors1 * sizeof(int));
  cudaMalloc(&d_matches, nDescriptors0 * sizeof(int));
  cudaMalloc(&d_scores, nDescriptors0 * sizeof(float));
  cudaMalloc(&d_scores1, nDescriptors1 * sizeof(float));
  // malloc for similarity matrix
  cudaMalloc(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float));
  cudaMalloc(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float));

  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  // we calculate the matches for the first image

  // similarity matrix will be nDescriptors0 x nDescriptors1
  dim3 threadsPerBlock2D(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid2D(
      (nDescriptors0 + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
      (nDescriptors1 + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
  similarityMatrixAndTransposeV2<<<blocksPerGrid2D, threadsPerBlock2D>>>(
      d_descriptors0, d_descriptors1, nDescriptors0, nDescriptors1, 128, d_sim,
      d_simT);

  cudaDeviceSynchronize();

  // matches
  find_nn<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores,
                                              nDescriptors0, nDescriptors1,
                                              ratio_thresh_sq);

  // transpose
  // otherside matches
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  find_nn<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                               nDescriptors1, nDescriptors0,
                                               ratio_thresh_sq);

  cudaDeviceSynchronize();

  // we check if the matches are mutual
  mutualCheck<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                  d_matches, nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpy(matches.data(), d_matches, nDescriptors0 * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(scores.data(), d_scores, nDescriptors0 * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_matches0);

  cudaFree(d_sim);
  cudaFree(d_simT);

  cudaFree(d_matches1);
  cudaFree(d_matches);
  cudaFree(d_scores);
  cudaFree(d_scores1);
}