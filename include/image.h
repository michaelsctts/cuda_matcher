// image.h
#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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

#endif  // IMAGE_H