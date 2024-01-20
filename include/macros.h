// macros.h

#ifndef MACROS_H
#define MACROS_H

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>

#define ASSERT(condition, message)                                     \
  if (!(condition)) {                                                  \
    std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
              << " line " << __LINE__ << ": " << message << std::endl; \
    exit(EXIT_FAILURE);                                                \
  }

// assert or return false
#define ASSERT_OR_RETURN(condition, message)                           \
  if (!(condition)) {                                                  \
    std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
              << " line " << __LINE__ << ": " << message << std::endl; \
    return false;                                                      \
  }

#define ASSERT_OR_VOID(condition, message)                             \
  if (!(condition)) {                                                  \
    std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
              << " line " << __LINE__ << ": " << message << std::endl; \
    return;                                                            \
  }

#define GET_ARG(arg) program.get<decltype(arg)>(#arg)
#define GET_DASH_ARG(arg) program.get<decltype(arg)>("--" #arg)

#define NOT_NULL(x) ASSERT_OR_RETURN(x != nullptr, #x " is null")

#define NOT_EMPTY(x) ASSERT_OR_RETURN(!x.empty(), #x " is empty")

#define NOT_ZERO(x) ASSERT_OR_RETURN(x != 0, #x " is zero")

#define NOT_NEGATIVE(x) ASSERT_OR_RETURN(x >= 0, #x " is negative")

static std::unordered_map<int, std::chrono::high_resolution_clock::time_point>
    startTimes;

#define TIMER_START(id)                                         \
  do {                                                          \
    startTimes[id] = std::chrono::high_resolution_clock::now(); \
  } while (0)

// cuda check call
#define CUDA_CHECK(call)                                                 \
  {                                                                      \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                           \
    }                                                                    \
  }

#define TIMER_END(id, message)                                         \
  do {                                                                 \
    auto endTime = std::chrono::high_resolution_clock::now();          \
    if (startTimes.find(id) != startTimes.end()) {                     \
      std::chrono::duration<double, std::milli> elapsedTime =          \
          endTime - startTimes[id];                                    \
      std::cout << (message) << " " << elapsedTime.count() << " ms\n"; \
    } else {                                                           \
      std::cout << "Timer ID " << id << " not found." << std::endl;    \
    }                                                                  \
  } while (0)

static std::unordered_map<int, std::thread> threadsMap;

// if i want to parallelize a function like sum(a,b,c,d) i can do:
// parallel_run(sum(a,b,c,d), 0)
#define parallel_run(call, th)          \
  do {                                  \
    threadsMap[th] = std::thread(call); \
  } while (0)
#define parallel_join(th)  \
  do {                     \
    threadsMap[th].join(); \
  } while (0)

#endif  // MACROS_H
