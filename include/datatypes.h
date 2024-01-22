// image.h
#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "matcher.h"

struct Image {
  int id;
  std::string name;

  // shape is size 2
  std::vector<float> shape;
  // keypoints are size 2 * num_keypoints
  std::vector<float> keypoints;
  // descriptors are size 128 * num_keypoints
  std::vector<float> descriptors;
  // scores are size num_keypoints
  std::vector<float> scores;

  float* d_descriptors;

  // matches[i] indicates the index of the match for the i-th descriptor
  // if matches[i] == -1, then the i-th descriptor has no match

  // matches are size 128 * num_keypoints
  std::vector<int> matches;

  // the score of the match for the i-th descriptor

  // match scores are size  128 * num_keypoints
  std::vector<float> match_scores;

  // getters because everything is flattened
  inline int getNumFeatures() { return scores.size(); }
  inline float getScore(int i) { return scores[i]; }
  inline float getKeypointX(int i) { return keypoints[2 * i]; }
  inline float getKeypointY(int i) { return keypoints[2 * i + 1]; }
  inline float* getKeypointXPtr(int i) { return &keypoints[2 * i]; }
  inline float* getKeypointYPtr(int i) { return &keypoints[2 * i + 1]; }
  inline float* getKeypointPtr(int i) { return &keypoints[2 * i]; }
  inline float* getDescriptorPtr(int i) { return &descriptors[128 * i]; }
  inline float* getScorePtr(int i) { return &scores[i]; }
  inline int* getMatchPtr(int i) { return &matches[128 * i]; }
  inline float* getMatchScorePtr(int i) { return &match_scores[128 * i]; }

  void allocateDescriptorsDevice() {
    allocateDescriptors(&d_descriptors, descriptors);
  }
  // TODO: setters

  // TODO: friend operator<< for printing
  //  print
  void print() {
    std::cout << "Image: " << name << std::endl;
    std::cout << "Shape: " << shape[0] << " " << shape[1] << std::endl;
    std::cout << "Num keypoints: " << keypoints.size() / 2 << std::endl;
    std::cout << "Num descriptors: " << descriptors.size() / 128 << std::endl;
    std::cout << "Num scores: " << scores.size() << std::endl;
    std::cout << "Num matches: " << matches.size() / 128 << std::endl;
    std::cout << "Num match scores: " << match_scores.size() / 128 << std::endl;
  }
};

struct Pair {
  Image* image0;
  Image* image1;
  std::vector<int> matches0;
  std::vector<int> matches1;
  std::vector<float> scores0;
  std::vector<float> scores1;
  std::vector<int> matches;
  std::vector<float> scores;

  friend std::ostream& operator<<(std::ostream& os, const Pair& pair) {
    os << pair.image0->name << " " << pair.image1->name << std::endl;
    for (int i = 0; i < pair.matches.size(); i++) {
      os << pair.matches[i] << " ";
    }
    os << std::endl;
    for (int i = 0; i < pair.scores.size(); i++) {
      os << pair.scores[i] << " ";
    }
    os << std::endl;
    return os;
  }
};

#endif  // IMAGE_H