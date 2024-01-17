// input.h

#ifndef INPUT_H
#define INPUT_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "image.h"

void read_features(std::string filename, std::vector<Image *> &images,
                   std::unordered_map<std::string, Image *> &images_map);

void read_pairs(std::string filename,
                std::vector<std::pair<Image *, Image *>> &pairs,
                std::unordered_map<std::string, Image *> &images_map);

#endif  // INPUT_H
