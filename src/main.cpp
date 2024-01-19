// main.cpp

#include <chrono>
#include <iostream>
#include <thread>

#include "input.h"
#include "matcher.h"
#include "math.h"

int run(int i, std::vector<Pair *> pairs) {
  for (int i = 0; i < pairs.size(); i++) {
    featureMatching(
        pairs[i]->image0->descriptors, pairs[i]->image1->descriptors,
        pairs[i]->matches, pairs[i]->scores, std::pow(0.95f, 2),
        pairs[i]->image0->getNumFeatures(), pairs[i]->image1->getNumFeatures());
  }
  return 0;
}

int main() {
  // TODO: argparsing
  std::string features_filename = "./features.txt";
  std::string pairs_filename = "./pairs.txt";
  int num_threads = 6;

  std::vector<Image *> images;
  std::unordered_map<std::string, Image *> images_map;
  read_features(features_filename, images, images_map);
  std::vector<Pair *> pairs;
  read_pairs(pairs_filename, pairs, images_map);

  std::vector<std::vector<Pair *>> pairs_parts(num_threads);
  for (int i = 0; i < pairs.size(); i++) {
    pairs_parts[i % num_threads].push_back(pairs[i]);
  }

  std::cout << "pairs: " << pairs.size() << std::endl;

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
  save_matches("./matches.txt", pairs);

  std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
  std::cout << "Mean per pair: " << elapsed.count() / pairs.size() << " ms\n";

  return 0;
}