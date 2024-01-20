#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

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
#pragma unroll 4
    for (int i = 0; i < descriptorDim; i += 4) {
      float4 vec0 = *((float4*)(descriptors0 + globalIdx + i));
      float4 vec1 = *((float4*)(descriptors1 + globalIdy + i));

      dotProduct += __fmaf_rn(vec0.x, vec1.x, 0.0f);
      dotProduct += __fmaf_rn(vec0.y, vec1.y, 0.0f);
      dotProduct += __fmaf_rn(vec0.z, vec1.z, 0.0f);
      dotProduct += __fmaf_rn(vec0.w, vec1.w, 0.0f);
    }

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
#pragma unroll 2
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
    float sim_nn0 = -1e30f;
    float sim_nn1 = -1e30f;
    int nearestNeighborIdx = -1;
    for (int i = 0; i < nDescriptors1; ++i) {
      float current_sim = sim[idx * nDescriptors1 + i];

      if (current_sim > sim_nn0) {
        sim_nn1 = sim_nn0;
        sim_nn0 = current_sim;
        nearestNeighborIdx = i;
      } else if (current_sim > sim_nn1) {
        sim_nn1 = current_sim;
      }
    }

    float dist_nn0 = 2 * (1 - sim_nn0);
    float dist_nn1 = 2 * (1 - sim_nn1);

    bool validMatch = (dist_nn0 <= ratio_thresh_sq * dist_nn1);

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? (sim_nn0 + 1) / 2.0f : 0.0f;
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

    bool validMatch = ((2 * (1 - sim0)) <= ratio_thresh_sq * (2 * (1 - sim1)));

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

__global__ void mutualCheckV2(int* matches0, const int* matches1,
                              int nDescriptors) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors) {
    int match1 = matches0[idx];
    int match2 = (match1 != -1) ? matches1[match1] : -1;
    matches0[idx] = (match2 == idx) ? match1 : -1;
  }
}

__global__ void mutualCheckV3(int* matches0, const int* matches1,
                              int nDescriptors) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors) {
    int match1 = matches0[idx];
    matches0[idx] = (match1 != -1 && matches1[match1] == idx) ? match1 : -1;
  }
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
  float* d_sim;
  float* d_simT;

  cudaMallocAsync(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float), 0);
  cudaMallocAsync(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float), 0);

  int *d_matches0, *d_matches1;
  float *d_scores0, *d_scores1;

  int threadsPerBlock = 32;
  dim3 threadsPerBlock2D(4, 4);

  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocksPerGrid2D(
      (nDescriptors0 + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
      (nDescriptors1 + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

  cudaDeviceSynchronize();

  cudaMallocAsync(&d_matches0, nDescriptors0 * sizeof(int), 0);
  cudaMallocAsync(&d_matches1, nDescriptors1 * sizeof(int), 0);
  cudaMallocAsync(&d_scores0, nDescriptors0 * sizeof(float), 0);
  cudaMallocAsync(&d_scores1, nDescriptors1 * sizeof(float), 0);

  similarityMatrixAndTransposeV2<<<blocksPerGrid2D, threadsPerBlock2D>>>(
      d_descriptors0, d_descriptors1, nDescriptors0, nDescriptors1, 128, d_sim,
      d_simT);

  cudaDeviceSynchronize();

  find_nnV2<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores0,
                                                nDescriptors0, nDescriptors1,
                                                ratio_thresh_sq);

  find_nnV2<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                                 nDescriptors1, nDescriptors0,
                                                 ratio_thresh_sq);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(scores.data(), d_scores0, nDescriptors0 * sizeof(float),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_sim, 0);
  cudaFreeAsync(d_simT, 0);
  cudaFreeAsync(d_scores1, 0);

  mutualCheckV3<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                    nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(matches.data(), d_matches0, nDescriptors0 * sizeof(int),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_matches0, 0);
  cudaFreeAsync(d_matches1, 0);
  cudaFreeAsync(d_scores0, 0);

  cudaDeviceSynchronize();
}