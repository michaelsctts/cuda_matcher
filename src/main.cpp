// main.cpp

#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include "input.h"
#include "macros.h"
#include "matcher.h"
#include "math.h"

int run(int i, std::vector<Pair *> pairs) {
  float ratio = std::pow(0.95, 2);
  for (int i = 0; i < pairs.size(); i++) {
    featureMatching(pairs[i]->image0->d_descriptors,
                    pairs[i]->image1->d_descriptors, pairs[i]->matches,
                    pairs[i]->scores, ratio, pairs[i]->image0->getNumFeatures(),
                    pairs[i]->image1->getNumFeatures());
  }
  return 0;
}

int runSingle(int i, Pair *pair, float ratio) {
  featureMatching(pair->image0->d_descriptors, pair->image1->d_descriptors,
                  pair->matches, pair->scores, ratio,
                  pair->image0->getNumFeatures(),
                  pair->image1->getNumFeatures());
  return 0;
}

int main(int argc, const char **argv) {
  // argparse
  argparse::ArgumentParser program("cuda_matcher");
  program.add_argument("-f", "features_path")
      .help("path to features file")
      .default_value(std::string("./features.txt"));
  program.add_argument("-p", "pairs_path")
      .help("path to pairs file")
      .default_value(std::string("./pairs.txt"));

  program.add_argument("-o", "output_path")
      .help("path to output file")
      .default_value(std::string("./matches.txt"));

  program.add_argument("-t", "num_threads")
      .help("number of threads")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  std::string features_path = GET_ARG(features_path);
  std::string pairs_path = GET_ARG(pairs_path);
  std::string output_path = GET_ARG(output_path);
  int num_threads = GET_ARG(num_threads);

  std::vector<Image *> images;
  std::unordered_map<std::string, Image *> images_map;
  std::vector<Pair *> pairs;
  std::vector<std::vector<Pair *>> pairs_parts(num_threads);
  std::vector<std::thread> threads;

  read_features(features_path, images, images_map);
  read_pairs(pairs_path, pairs, images_map);

  for (int i = 0; i < pairs.size(); i++) {
    pairs_parts[i % num_threads].push_back(pairs[i]);
  }

  std::cout << "pairs: " << pairs.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  float ratio = std::pow(0.95, 2);

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(run, i, pairs_parts[i]));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  // TODO: save matches and scores
  save_matches(output_path, pairs);

  count_matches(pairs);

  std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
  std::cout << "Mean per pair: " << elapsed.count() / pairs.size() << " ms\n";

  return 0;
}