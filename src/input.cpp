// input.cpp

#include "input.h"

void read_features(std::string filename, std::vector<Image *> &images,
                   std::unordered_map<std::string, Image *> &images_map) {
  std::ifstream file(filename);

  std::string line;

  while (std::getline(file, line)) {
    std::string name;
    std::stringstream ss(line);
    int id;
    ss >> id;
    ss >> name;

    std::vector<float> shape;
    float value;
    for (int i = 0; i < 2; i++) {
      ss >> value;
      shape.push_back(value);
    }

    std::getline(file, line);
    ss = std::stringstream(line);
    int num_keypoints;
    ss >> num_keypoints;

    std::vector<float> keypoints(2 * num_keypoints);
    std::getline(file, line);
    ss = std::stringstream(line);
    for (int i = 0; i < 2 * num_keypoints; i++) {
      ss >> keypoints[i];
    }

    // TODO: change file order, should be descriptors[i*num_keypoints + j]
    // lo hice al reves xd

    std::vector<float> descriptors(128 * num_keypoints);
    for (int i = 0; i < 128; i++) {
      std::getline(file, line);
      ss = std::stringstream(line);
      for (int j = 0; j < num_keypoints; j++) {
        ss >> descriptors[i + j * 128];
        // ss >> descriptors[j + i * num_keypoints];
      }
    }

    std::vector<float> scores(num_keypoints);
    std::getline(file, line);
    ss = std::stringstream(line);
    for (int i = 0; i < num_keypoints; i++) {
      ss >> scores[i];
    }

    // allocate here so we dont allocate on matching
    std::vector<int> matches(num_keypoints, -1);
    std::vector<float> match_scores(num_keypoints);

    Image *image = new Image();

    image->id = id;
    image->name = name;
    image->shape = shape;
    image->keypoints = keypoints;
    image->descriptors = descriptors;
    image->scores = scores;
    image->matches = matches;
    image->match_scores = match_scores;

    images.push_back(image);
    images_map[name] = image;

    image->allocateDescriptorsDevice();
  }

  file.close();
}

void read_pairs(std::string filename, std::vector<Pair *> &pairs,
                std::unordered_map<std::string, Image *> &images_map) {
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string name0, name1;

    ss >> name0 >> name1;

    Image *image0 = images_map[name0];
    Image *image1 = images_map[name1];

    if (image0 == nullptr || image1 == nullptr) {
      std::cout << "Error: image not found" << std::endl;
      exit(1);
    }
    Pair *pair = new Pair();
    pair->image0 = image0;
    pair->image1 = image1;
    pair->matches = std::vector<int>(image0->getNumFeatures(), -1);
    pair->scores = std::vector<float>(image0->getNumFeatures());
    // pair->scores0 = std::vector<float>(image0->getNumFeatures());
    // pair->scores1 = std::vector<float>(image1->getNumFeatures());
    pairs.push_back(pair);
  }

  file.close();
}

void save_matches(std::string filename, std::vector<Pair *> &pairs) {
  std::ofstream file(filename);

  for (int i = 0; i < pairs.size(); i++) {
    file << *pairs[0] << std::endl;
  }
  file.close();
}

void count_matches(std::vector<Pair *> &pairs) {
  int total_keypoints = 0;
  int total_matches = 0;
  for (int i = 0; i < pairs.size(); i++) {
    total_keypoints += pairs[i]->image0->getNumFeatures();
    for (int j = 0; j < pairs[i]->image0->getNumFeatures(); j++) {
      if (pairs[i]->matches[j] != -1) {
        total_matches++;
      }
    }
  }
  std::cout << "sexo" << std::endl;
  std::cout << "Total matches: " << total_matches << std::endl;
  std::cout << "Total keypoints: " << total_keypoints << std::endl;
}
