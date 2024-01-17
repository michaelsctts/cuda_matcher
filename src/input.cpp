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
      }
    }

    std::vector<float> scores(num_keypoints);
    std::getline(file, line);
    ss = std::stringstream(line);
    for (int i = 0; i < num_keypoints; i++) {
      ss >> scores[i];
    }

    // allocate here so we dont allocate on matching
    std::vector<int> matches(128 * num_keypoints, -1);
    std::vector<float> match_scores(128 * num_keypoints);

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
  }

  file.close();
}

void read_pairs(std::string filename,
                std::vector<std::pair<Image *, Image *>> &pairs,
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
    pairs.push_back(std::make_pair(image0, image1));
  }

  file.close();
}