#include <cublas_v2.h>

#include <chrono>
#include <iostream>
#include <vector>

#ifdef PYBIND
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#include <cuda_runtime.h>

#ifdef FP16
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#define TILE_WIDTH 16

#define FLOAT_LOWEST -340282346638528859811704183484516925440.0

__device__ __forceinline__ float multiply(float a, float b) {
  return __fmul_rn(a, b);
}

__device__ __forceinline__ float sum(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float subtract(float a, float b) {
  return __fsub_rn(a, b);
}

__device__ __forceinline__ bool greater(float a, float b) { return a > b; }

__device__ __forceinline__ bool greaterEqual(float a, float b) {
  return a >= b;
}

__device__ __forceinline__ bool less(float a, float b) { return a < b; }

__device__ __forceinline__ bool lessEqual(float a, float b) { return a <= b; }

__device__ __forceinline__ bool equal(float a, float b) { return a == b; }

#ifdef FP16
__device__ __forceinline__ half multiply(half a, half b) {
  return __hmul(a, b);
}
__device__ __forceinline__ half sum(half a, half b) { return __hadd(a, b); }

__device__ __forceinline__ half subtract(half a, half b) {
  return __hsub(a, b);
}

__device__ __forceinline__ half2 multiply(half2 a, half2 b) {
  return __hmul2(a, b);
}
__device__ __forceinline__ half2 sum(half2 a, half2 b) { return __hadd2(a, b); }

__device__ __forceinline__ half2 subtract(half2 a, half2 b) {
  return __hsub2(a, b);
}

#endif

// SimilarityMatrixAndTransposeV2 is faster but this is kept for reference
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

// TODO: check if pre-setting to 0.0 is faster than checking for overflow in
// shared memory
__global__ void matrixMultiplySharedLegacy(const float* A, const float* B,
                                           float* C, float* CT, int numARows,
                                           int numAColumns, int numBRows,
                                           int numBColumns, int numCRows,
                                           int numCColumns) {
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  float Cvalue = 0.0;

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
      Cvalue = sum(Cvalue, multiply(sA[threadIdx.y][j], sB[j][threadIdx.x]));
    }
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
    CT[Col * numCRows + Row] = Cvalue;
  }
}

template <typename T>
__global__ void matrixMultiplyShared(const T* A, const T* B, T* C, T* CT,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ T sA[TILE_WIDTH][TILE_WIDTH];
  __shared__ T sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  T Cvalue = 0.0;
  int numTiles = ((numAColumns + TILE_WIDTH - 1) / TILE_WIDTH);

  for (int ph = 0; ph < numTiles; ++ph) {
    int aRow = Row;
    int aCol = threadIdx.x + ph * TILE_WIDTH;
    int bRow = threadIdx.y + ph * TILE_WIDTH;
    int bCol = Col;

    sA[threadIdx.y][threadIdx.x] = (aRow < numARows && aCol < numAColumns)
                                       ? A[aRow * numAColumns + aCol]
                                       : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < numBRows && bCol < numBColumns)
                                       ? B[bRow * numBColumns + bCol]
                                       : 0.0f;

    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; ++j) {
      Cvalue = sum(Cvalue, multiply(sA[threadIdx.y][j], sB[j][threadIdx.x]));
    }
    __syncthreads();
  }

  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
    CT[Col * numCRows + Row] = Cvalue;
  }
}

// tiled matmul is faster, tiling this might be faster
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

template <typename T>
__global__ void transpose(const T* in, T* out, int n, int m) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < n && idy < m) {
    out[idy * n + idx] = in[idx * m + idy];
  }
}

__global__ void find_nnlegacy(const float* sim, int* matches, float* scores,
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

__global__ void find_nnV2legacy(const float* sim, int* matches, float* scores,
                                const int nDescriptors0,
                                const int nDescriptors1,
                                const float ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    float sim_nn0 = -1e30f;
    float sim_nn1 = -1e30f;
    int nearestNeighborIdx = -1;
#pragma unroll 4
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
    scores[idx] = (validMatch) ? (sim_nn0 + 1) * 0.5f : 0.0f;
  }
}

template <typename T>
__global__ void find_nn(const T* sim, int* matches, T* scores,
                        const int nDescriptors0, const int nDescriptors1,
                        const T ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    T sim_nn0 = T(-1e15);
    T sim_nn1 = T(-1e15);
    int nearestNeighborIdx = -1;
#pragma unroll 32
    for (int i = 0; i < nDescriptors1; ++i) {
      T current_sim = sim[idx * nDescriptors1 + i];

      if (greater(current_sim, sim_nn0)) {
        sim_nn1 = sim_nn0;
        sim_nn0 = current_sim;
        nearestNeighborIdx = i;
      } else if (greater(current_sim, sim_nn1)) {
        sim_nn1 = current_sim;
      }
    }
    T dist_nn0 = multiply(T(2), subtract(T(1), sim_nn0));
    T dist_nn1 = multiply(T(2), subtract(T(1), sim_nn1));

    bool validMatch = lessEqual(dist_nn0, multiply(ratio_thresh_sq, dist_nn1));

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] =
        (validMatch) ? (multiply(sum(sim_nn0, T(1.0)), T(0.5))) : T(0.0);
  }
}

__global__ void mutualCheck(int* matches0, const int* matches1,
                            int nDescriptors) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors) {
    int match1 = matches0[idx];
    int match2 = (match1 != -1) ? matches1[match1] : -1;
    matches0[idx] = (match2 == idx) ? match1 : -1;
  }
}

__global__ void mutualCheckFast(int* matches0, const int* matches1,
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

void deallocateDescriptors(float* d_descriptors) { cudaFree(d_descriptors); }

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

  find_nnlegacy<<<blocksPerGrid, threadsPerBlock>>>(
      d_sim, d_matches0, d_scores0, nDescriptors0, nDescriptors1,
      ratio_thresh_sq);

  find_nnlegacy<<<blocksPerGridT, threadsPerBlock>>>(
      d_simT, d_matches1, d_scores1, nDescriptors1, nDescriptors0,
      ratio_thresh_sq);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(scores.data(), d_scores0, nDescriptors0 * sizeof(float),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_sim, 0);
  cudaFreeAsync(d_simT, 0);
  cudaFreeAsync(d_scores1, 0);

  mutualCheckFast<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
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

  dim3 threadsPerBlock2Ddt(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid2Ddt(
      (nDescriptors1 + threadsPerBlock2Ddt.x - 1) / threadsPerBlock2Ddt.x,
      (128 + threadsPerBlock2Ddt.y - 1) / threadsPerBlock2Ddt.y);

  cudaDeviceSynchronize();

  transpose<<<blocksPerGrid2Ddt, threadsPerBlock2Ddt>>>(
      d_descriptors1, d_descriptors1T, nDescriptors1, 128);

  dim3 threadsPerBlock2Dmult(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid2Dmult((nDescriptors1 + TILE_WIDTH - 1) / TILE_WIDTH,
                           (nDescriptors0 + TILE_WIDTH - 1) / TILE_WIDTH);

  cudaDeviceSynchronize();

  matrixMultiplyShared<<<blocksPerGrid2Dmult, threadsPerBlock2Dmult>>>(
      d_descriptors0, d_descriptors1T, d_sim, d_simT, nDescriptors0, 128, 128,
      nDescriptors1, nDescriptors0, nDescriptors1);

  int threadsPerBlock = TILE_WIDTH;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  cudaDeviceSynchronize();

  find_nn<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores0,
                                              nDescriptors0, nDescriptors1,
                                              ratio_thresh_sq);

  find_nn<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                               nDescriptors1, nDescriptors0,
                                               ratio_thresh_sq);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(scores.data(), d_scores0, nDescriptors0 * sizeof(float),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_descriptors1T, 0);
  cudaFreeAsync(d_sim, 0);
  cudaFreeAsync(d_simT, 0);
  cudaFreeAsync(d_scores1, 0);

  mutualCheckFast<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                      nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(matches.data(), d_matches0, nDescriptors0 * sizeof(int),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_matches0, 0);
  cudaFreeAsync(d_matches1, 0);
  cudaFreeAsync(d_scores0, 0);

  // cudaDeviceSynchronize();
}

#ifdef FP16
__global__ void matrixMultiplyShared(const half* A, const half* B, half* C,
                                     half* CT, int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ half sA[TILE_WIDTH][TILE_WIDTH];
  __shared__ half sB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  half Cvalue = 0.0;
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
      // sm53 and above
      // lower sm must convert to float and back
      Cvalue += __hmul(sA[threadIdx.y][j], sB[j][threadIdx.x]);
    }
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
    CT[Col * numCRows + Row] = Cvalue;
  }
}

__global__ void find_nnV2(const half* sim, int* matches, half* scores,
                          const int nDescriptors0, const int nDescriptors1,
                          const half ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    half sim_nn0 = __float2half(-1e15f);
    half sim_nn1 = __float2half(-1e15f);
    int nearestNeighborIdx = -1;

    for (int i = 0; i < nDescriptors1; ++i) {
      half current_sim = sim[idx * nDescriptors1 + i];

      if (__hgt(current_sim, sim_nn0)) {
        sim_nn1 = sim_nn0;
        sim_nn0 = current_sim;
        nearestNeighborIdx = i;
      } else if (__hgt(current_sim, sim_nn1)) {
        sim_nn1 = current_sim;
      }
    }

    half dist_nn0 =
        __hmul(__float2half(2.0f), __hsub(__float2half(1.0f), sim_nn0));
    half dist_nn1 =
        __hmul(__float2half(2.0f), __hsub(__float2half(1.0f), sim_nn1));

    bool validMatch = __hle(dist_nn0, __hmul(ratio_thresh_sq, dist_nn1));

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? __hdiv(__hadd(sim_nn0, __float2half(1.0f)),
                                        __float2half(2.0f))
                               : __float2half(0.0f);
  }
}

__global__ void float2half(const float* in, half* out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}

__global__ void half2float(const half* in, float* out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) {
    out[idx] = __half2float(in[idx]);
  }
}

void featureMatchingHalf(const half* d_descriptors0, const half* d_descriptors1,
                         std::vector<int>& matches, std::vector<float>& scores,
                         float ratio_thresh_sq, int nDescriptors0,
                         int nDescriptors1) {
  half* d_sim;
  half* d_simT;

  cudaMallocAsync(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(half), 0);
  cudaMallocAsync(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(half), 0);

  int *d_matches0, *d_matches1;
  half *d_scores0, *d_scores1;

  cudaMallocAsync(&d_matches0, nDescriptors0 * sizeof(int), 0);
  cudaMallocAsync(&d_matches1, nDescriptors1 * sizeof(int), 0);
  cudaMallocAsync(&d_scores0, nDescriptors0 * sizeof(half), 0);
  cudaMallocAsync(&d_scores1, nDescriptors1 * sizeof(half), 0);

  half* d_descriptors1T;
  cudaMalloc(&d_descriptors1T, nDescriptors1 * 128 * sizeof(half));

  dim3 threadsPerBlock2Ddt(8, 8);
  dim3 blocksPerGrid2Ddt(
      (nDescriptors1 + threadsPerBlock2Ddt.x - 1) / threadsPerBlock2Ddt.x,
      (128 + threadsPerBlock2Ddt.y - 1) / threadsPerBlock2Ddt.y);

  cudaDeviceSynchronize();

  transpose<half><<<blocksPerGrid2Ddt, threadsPerBlock2Ddt>>>(
      d_descriptors1, d_descriptors1T, nDescriptors1, 128);

  dim3 threadsPerBlock2Dmult(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid2Dmult((nDescriptors1 / TILE_WIDTH + 1),
                           (nDescriptors0 / TILE_WIDTH + 1));

  matrixMultiplyShared<<<blocksPerGrid2Dmult, threadsPerBlock2Dmult>>>(
      d_descriptors0, d_descriptors1T, d_sim, d_simT, nDescriptors0, 128, 128,
      nDescriptors1, nDescriptors0, nDescriptors1);

  int threadsPerBlock = TILE_WIDTH;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridT = (nDescriptors1 + threadsPerBlock - 1) / threadsPerBlock;

  cudaDeviceSynchronize();

  find_nnV2<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores0,
                                                nDescriptors0, nDescriptors1,
                                                ratio_thresh_sq);

  find_nnV2<<<blocksPerGridT, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
                                                 nDescriptors1, nDescriptors0,
                                                 ratio_thresh_sq);

  cudaDeviceSynchronize();

  // copy scores to float
  float* scoresFloat;
  cudaMalloc(&scoresFloat, nDescriptors0 * sizeof(float));
  half2float<<<blocksPerGrid, threadsPerBlock>>>(d_scores0, scoresFloat,
                                                 nDescriptors0);

  cudaMemcpyAsync(scores.data(), scoresFloat, nDescriptors0 * sizeof(float),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_descriptors1T, 0);
  cudaFreeAsync(d_sim, 0);
  cudaFreeAsync(d_simT, 0);
  cudaFreeAsync(d_scores1, 0);
  cudaFreeAsync(scoresFloat, 0);

  mutualCheckFast<<<blocksPerGrid, threadsPerBlock>>>(d_matches0, d_matches1,
                                                      nDescriptors0);

  cudaDeviceSynchronize();

  cudaMemcpyAsync(matches.data(), d_matches0, nDescriptors0 * sizeof(int),
                  cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_matches0, 0);
  cudaFreeAsync(d_matches1, 0);
  cudaFreeAsync(d_scores0, 0);

  cudaDeviceSynchronize();
}

// wrapper for half precision from float
void featureMatchingHalf(float* d_descriptors0, float* d_descriptors1,
                         std::vector<int>& matches, std::vector<float>& scores,
                         float ratio_thresh_sq, int nDescriptors0,
                         int nDescriptors1) {
  half *d_descriptors0_half, *d_descriptors1_half;

  cudaMalloc(&d_descriptors0_half, nDescriptors0 * 128 * sizeof(half));
  cudaMalloc(&d_descriptors1_half, nDescriptors1 * 128 * sizeof(half));

  float2half<<<(nDescriptors0 * 128 + 127) / 128, 128>>>(
      d_descriptors0, d_descriptors0_half, nDescriptors0 * 128);
  float2half<<<(nDescriptors1 * 128 + 127) / 128, 128>>>(
      d_descriptors1, d_descriptors1_half, nDescriptors1 * 128);

  featureMatchingHalf(d_descriptors0_half, d_descriptors1_half, matches, scores,
                      ratio_thresh_sq, nDescriptors0, nDescriptors1);

  cudaFree(d_descriptors0_half);
  cudaFree(d_descriptors1_half);
}

#endif

#ifdef PYBIND

namespace py = pybind11;

void featureMatchingWrapper(py::array_t<float> descriptors0,
                            py::array_t<float> descriptors1,
                            float ratio_thresh_sq, std::vector<int>& matches,
                            std::vector<float>& scores) {
  py::buffer_info buf0 = descriptors0.request();
  py::buffer_info buf1 = descriptors1.request();

  if (buf0.ndim != 2 || buf1.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be two");
  }

  if (buf0.shape[1] != 128 || buf1.shape[1] != 128) {
    throw std::runtime_error("Number of columns must be 128");
  }

  int nDescriptors0 = buf0.shape[0];
  int nDescriptors1 = buf1.shape[0];

  float* d_descriptors0;
  float* d_descriptors1;

  cudaMalloc(&d_descriptors0, nDescriptors0 * 128 * sizeof(float));
  cudaMalloc(&d_descriptors1, nDescriptors1 * 128 * sizeof(float));

  cudaMemcpy(d_descriptors0, buf0.ptr, nDescriptors0 * 128 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_descriptors1, buf1.ptr, nDescriptors1 * 128 * sizeof(float),
             cudaMemcpyHostToDevice);

  featureMatching(d_descriptors0, d_descriptors1, matches, scores,
                  ratio_thresh_sq, nDescriptors0, nDescriptors1);

  cudaFree(d_descriptors0);
  cudaFree(d_descriptors1);
}

// pybind
PYBIND11_MODULE(feature_matching, m) {
  m.doc() = "Feature matching module";

  m.def("featureMatching", &featureMatchingWrapper,
        "A function which performs feature matching");
}

#endif