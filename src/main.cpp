// main.cpp

#include <chrono>
#include <iostream>
#include <thread>

#include "input.h"
#include "matcher.h"

//
int run(int i, std::vector<std::pair<Image *, Image *>> pairs) {
  for (int i = 0; i < pairs.size(); i++) {
    featureMatching(pairs[i].first->descriptors, pairs[i].second->descriptors,
                    pairs[i].first->matches, pairs[i].first->match_scores, 0.95,
                    -1, pairs[i].first->scores.size());
  }
  return 0;
}

int main() {
  // TODO: argparsing
  std::string features_filename = "./features.txt";
  std::string pairs_filename = "./pairs.txt";
  int num_threads = 4;

  std::vector<Image *> images;
  std::unordered_map<std::string, Image *> images_map;
  read_features(features_filename, images, images_map);
  std::vector<std::pair<Image *, Image *>> pairs;
  read_pairs(pairs_filename, pairs, images_map);

  std::cout << "pairs: " << pairs.size() << std::endl;

  std::vector<std::vector<std::pair<Image *, Image *>>> pairs_parts(
      num_threads);
  for (int i = 0; i < pairs.size(); i++) {
    pairs_parts[i % num_threads].push_back(pairs[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(run, i, pairs_parts[i]));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  // TODO: save matches and scores

  std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
  std::cout << "Mean per pair: " << elapsed.count() / pairs.size() << " ms\n";

  return 0;
}