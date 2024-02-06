// operations.cuh

#pragma once

#ifdef FP16
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

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

__device__ __forceinline__ bool greater(half a, half b) { return __hgt(a, b); }

__device__ __forceinline__ bool greaterEqual(half a, half b) {
  return __hge(a, b);
}

__device__ __forceinline__ bool less(half a, half b) { return __hlt(a, b); }

__device__ __forceinline__ bool lessEqual(half a, half b) {
  return __hle(a, b);
}

__device__ __forceinline__ bool equal(half a, half b) { return __heq(a, b); }

__device__ __forceinline__ half2 multiply(half2 a, half2 b) {
  return __hmul2(a, b);
}
__device__ __forceinline__ half2 sum(half2 a, half2 b) { return __hadd2(a, b); }

__device__ __forceinline__ half2 subtract(half2 a, half2 b) {
  return __hsub2(a, b);
}

__device__ __forceinline__ bool greater(half2 a, half2 b) {
  return __hgt2(a, b);
}

__device__ __forceinline__ bool greaterEqual(half2 a, half2 b) {
  return __hge2(a, b);
}

__device__ __forceinline__ bool less(half2 a, half2 b) { return __hlt2(a, b); }

__device__ __forceinline__ bool lessEqual(half2 a, half2 b) {
  return __hle2(a, b);
}

__device__ __forceinline__ bool equal(half2 a, half2 b) { return __heq2(a, b); }

#endif
