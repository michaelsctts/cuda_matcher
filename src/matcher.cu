#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void similiarityMatrix(const float* descriptors0,
                                  const float* descriptors1, int nDescriptors0,
                                  int nDescriptors1, int descriptorDim,
                                  float* sim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    float distance = 0.0f;
    for (int i = 0; i < descriptorDim; ++i) {
      float diff = descriptors0[idx * descriptorDim + i] -
                   descriptors1[idy * descriptorDim + i];
      //
      distance += diff * diff;

      sim[idx * nDescriptors1 + idy] = distance;
    }
  }
}

// einsum "ij,jk->jk"
__global__ void einsum(const float* descriptors0, const float* descriptors1,
                       int nDescriptors0, int nDescriptors1, int descriptorDim,
                       float* sim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    float dotProduct = 0.0f;
    for (int i = 0; i < descriptorDim; ++i) {
      dotProduct += descriptors0[idx * descriptorDim + i] *
                    descriptors1[idy * descriptorDim + i];
    }

    sim[idx * nDescriptors1 + idy] = dotProduct;
  }
}

// mirror to:
// sim = torch.einsum("bdn,bdm->bnm", data["descriptors0"],
// data["descriptors1"])
__global__ void similiarityMatrixAndTranspose(
    const float* descriptors0, const float* descriptors1, int nDescriptors0,
    int nDescriptors1, int descriptorDim, float* sim, float* simT) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    float dotProduct = 0.0f;
    for (int i = 0; i < descriptorDim; ++i) {
      dotProduct += descriptors0[idx * descriptorDim + i] *
                    descriptors1[idy * descriptorDim + i];
    }

    // Store the result in both sim and its transpose simT
    sim[idx * nDescriptors1 + idy] = dotProduct;
    simT[idy * nDescriptors0 + idx] = dotProduct;
  }
}

__global__ void transposeSimilarityMatrix(const float* sim, float* simT,
                                          int nDescriptors0,
                                          int nDescriptors1) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < nDescriptors0 && idy < nDescriptors1) {
    simT[idy * nDescriptors0 + idx] = sim[idx * nDescriptors1 + idy];
  }
}

/*
mirror this:
def find_nn(sim, ratio_thresh, distance_thresh):
    //sim is einsum("bdn,bdm->bnm", data["descriptors0"], data["descriptors1"])
    // so its equivalent in cuda c++ is:



    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores
*/

// device funct to find max and second max

__global__ void find_nn(const float* sim, int* matches, float* scores,
                        const int nDescriptors0, const int nDescriptors1,
                        const float ratio_thresh_sq) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < nDescriptors0) {
    // float sim_nn0 will store the greatest similarity, so it will initialize
    // with the lowest value
    float sim_nn0 = -100000.0f;
    float sim_nn1 = -100000.0f;
    int nearestNeighborIdx = -1;
    for (int i = 0; i < nDescriptors1; ++i) {
      // float sim_value = sim[idx * nDescriptors1 + i];
      if (sim[idx * nDescriptors1 + i] > sim_nn0) {
        sim_nn1 = sim_nn0;
        sim_nn0 = sim[idx * nDescriptors1 + i];
        nearestNeighborIdx = i;
      } else if (sim[idx * nDescriptors1 + i] > sim_nn1) {
        sim_nn1 = sim[idx * nDescriptors1 + i];
      }
    }

    // distnn = 2 * (1 - simnn)
    float dist_nn0 = 2 * (1 - sim_nn0);
    float dist_nn1 = 2 * (1 - sim_nn1);

    bool validMatch = true;
    if (ratio_thresh_sq > 0) {
      validMatch = (dist_nn0 <= ratio_thresh_sq * dist_nn1);
    }

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
    float minDistance = FLT_MAX;
    int nearestNeighborIdx = -1;

    for (int i = 0; i < nDescriptors1; ++i) {
      float distance = 0.0f;
      for (int j = 0; j < 128; ++j) {
        float diff = descriptors0[idx * 128 + j] - descriptors1[i * 128 + j];
        distance += diff;
      }

      if (distance < minDistance) {
        minDistance = distance;
        nearestNeighborIdx = i;
      }
    }

    // float distance_thresh_sq_check = distance_thresh_sq * distance_thresh_sq;
    bool validMatch = true;
    if (ratio_thresh_sq > 0) {
      float secondDistance = FLT_MAX;
      for (int i = 0; i < nDescriptors1; ++i) {
        if (i != nearestNeighborIdx) {
          float distance = 0.0f;
          for (int j = 0; j < 128; ++j) {
            float diff =
                descriptors0[idx * 128 + j] - descriptors1[i * 128 + j];
            distance += diff;
          }
          secondDistance =
              (distance < secondDistance) ? distance : secondDistance;
        }
      }
      validMatch = (minDistance <= ratio_thresh_sq * secondDistance);
    }

    matches[idx] = (validMatch) ? nearestNeighborIdx : -1;
    scores[idx] = (validMatch) ? (1.0f - minDistance) / 2.0f : 0.0f;
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
  if (descriptors0.size() != nDescriptors0 * 128) {
    std::cout << "Error: nDescriptors0 * 128 != descriptors0.size()"
              << std::endl;
    return;
  }
  if (descriptors1.size() != nDescriptors1 * 128) {
    std::cout << "Error: nDescriptors1 * 128 != descriptors1.size()"
              << std::endl;
    return;
  }
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
  cudaMalloc(&d_scores1, nDescriptors0 * sizeof(float));
  // malloc for similarity matrix
  float* d_sim;
  float* d_simT;
  cudaMalloc(&d_sim, nDescriptors0 * nDescriptors1 * sizeof(float));
  cudaMalloc(&d_simT, nDescriptors0 * nDescriptors1 * sizeof(float));

  int threadsPerBlock = 32;
  int blocksPerGrid = (nDescriptors0 + threadsPerBlock - 1) / threadsPerBlock;
  // we calculate the matches for the first image

  // similarity matrix will be nDescriptors0 x nDescriptors1
  dim3 threadsPerBlock2D(32, 32);
  dim3 blocksPerGrid2D(
      (nDescriptors0 + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
      (nDescriptors1 + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
  similiarityMatrixAndTranspose<<<blocksPerGrid2D, threadsPerBlock2D>>>(
      d_descriptors0, d_descriptors1, nDescriptors0, nDescriptors1, 128, d_sim,
      d_simT);

  cudaDeviceSynchronize();

  // matches
  find_nn<<<blocksPerGrid, threadsPerBlock>>>(d_sim, d_matches0, d_scores,
                                              nDescriptors0, nDescriptors1,
                                              ratio_thresh_sq);
  cudaDeviceSynchronize();

  // transpose
  // otherside matches
  find_nn<<<blocksPerGrid, threadsPerBlock>>>(d_simT, d_matches1, d_scores1,
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
