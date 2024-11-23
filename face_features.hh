// face_features.h
#ifndef FACE_FEATURES_H
#define FACE_FEATURES_H

#include <vector>
#include <string>
#include <dlib/dnn.h>
#include "db_utils.hh"

std::vector<float> extract_face_features(const dlib::matrix<dlib::rgb_pixel>& img);
void initialize_models();

#endif // FACE_FEATURES_H
