#include <cublas_v2.h>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>
#define BLOCK_SIZE 16
#define TILE_WIDTH 16

#define FLOAT_LOWEST -340282346638528859811704183484516925440.0

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

__global__ void matrixMultiplyShared(const float* A, const float* B, float* C,
                                     float* CT, int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  float Cvalue = 0.0;
  sA[threadIdx.y][threadIdx.x] = 0.0;
  sB[threadIdx.y][threadIdx.x] = 0.0;

  for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
    if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
      sA[threadIdx.y][threadIdx.x] =
          A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
      sB[threadIdx.y][threadIdx.x] =
          B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; ++j) {
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
    CT[Col * numCRows + Row] = Cvalue;
  }
}

void simCublas(const float* descriptors0, const float* descriptors1,
               int nDescriptors0, int nDescriptors1, int descriptorDim,
               float* sim) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t status =
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nDescriptors1,
                  nDescriptors0, 128, &alpha, descriptors1, nDescriptors1,
                  descriptors0, nDescriptors0, &beta, sim, nDescriptors1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "cublasSgemm failed" << std::endl;
  }

  cublasDestroy(handle);
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

__global__ void similarityMatrixFast(const float* descriptors0,
                                     const float* descriptors1,
                                     int nDescriptors0, int nDescriptors1,
                                     int descriptorDim, float* sim) {
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
  }
}

__global__ void transposeSim(const float* sim, float* simT, int nDescriptors0,
                             int nDescriptors1) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    simT[idy * nDescriptors0 + idx] = sim[idx * nDescriptors1 + idy];
  }
}

__global__ void transposeDescriptors(const float* descriptors,
                                     float* descriptorsT, int nDescriptors,
                                     int descriptorDim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors && idy < descriptorDim) {
    descriptorsT[idy * nDescriptors + idx] =
        descriptors[idx * descriptorDim + idy];
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

void featureMatchingLegacy(const float* d_descriptors0,
                           const float* d_descriptors1,
                           std::vector<int>& matches,
                           std::vector<float>& scores, float ratio_thresh_sq,
                           int nDescriptors0, int nDescriptors1) {
  float* d_sim;
  float* d_simT;

  cudaMallocAsync(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float), 0);
  cudaMallocAsync(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float), 0);

  int *d_matches0, *d_matches1;
  float *d_scores0, *d_scores1;

  int threadsPerBlock = 128;
  dim3 threadsPerBlock2D(8, 8);

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

void featureMatching(const float* d_descriptors0, const float* d_descriptors1,
                     std::vector<int>& matches, std::vector<float>& scores,
                     float ratio_thresh_sq, int nDescriptors0,
                     int nDescriptors1) {
  float* d_sim;
  float* d_simT;

  cudaMallocAsync(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float), 0);
  cudaMallocAsync(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float), 0);

  int *d_matches0, *d_matches1;
  float *d_scores0, *d_scores1;

  cudaDeviceSynchronize();

  cudaMallocAsync(&d_matches0, nDescriptors0 * sizeof(int), 0);
  cudaMallocAsync(&d_matches1, nDescriptors1 * sizeof(int), 0);
  cudaMallocAsync(&d_scores0, nDescriptors0 * sizeof(float), 0);
  cudaMallocAsync(&d_scores1, nDescriptors1 * sizeof(float), 0);

  float* d_descriptors1T;
  cudaMalloc(&d_descriptors1T, nDescriptors1 * 128 * sizeof(float));

  cudaDeviceSynchronize();

  dim3 threadsPerBlock2Ddt(8, 8);
  dim3 blocksPerGrid2Ddt(
      (nDescriptors1 + threadsPerBlock2Ddt.x - 1) / threadsPerBlock2Ddt.x,
      (128 + threadsPerBlock2Ddt.y - 1) / threadsPerBlock2Ddt.y);

  transposeDescriptors<<<blocksPerGrid2Ddt, threadsPerBlock2Ddt>>>(
      d_descriptors1, d_descriptors1T, nDescriptors1, 128);

  cudaDeviceSynchronize();

  dim3 threadsPerBlock2Dmult(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid2Dmult((nDescriptors1 / TILE_WIDTH + 1),
                           (nDescriptors0 / TILE_WIDTH + 1));

  matrixMultiplyShared<<<blocksPerGrid2Dmult, threadsPerBlock2Dmult>>>(
      d_descriptors0, d_descriptors1T, d_sim, d_simT, nDescriptors0, 128, 128,
      nDescriptors1, nDescriptors0, nDescriptors1);

  cudaDeviceSynchronize();

  int threadsPerBlock = TILE_WIDTH;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  find_nnV2<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores0,
                                                nDescriptors0, nDescriptors1,
                                                ratio_thresh_sq);

  find_nnV2<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                                 nDescriptors1, nDescriptors0,
                                                 ratio_thresh_sq);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(scores.data(), d_scores0, nDescriptors0 * sizeof(float),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_descriptors1T, 0);
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