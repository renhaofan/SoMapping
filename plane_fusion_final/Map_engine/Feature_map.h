

#pragma once

//
#include "OurLib/My_matrix.h"
//
#include "Feature_detector/Feature_detector.h"
#include "Map_engine/voxel_definition.h"
#include "OurLib/My_pose.h"

// OpenCV
#include <cv.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// DBoW3
#include <DBoW3/DBoW3.h>

//
#include <iostream>
#include <unordered_map>
#include <vector>

//
typedef struct struct_Model_keypoint {
  //
  My_Type::Vector3f point;
  //
  float weight;
  //
  bool is_valid;

  //
  int non_observe_counter;
  int observe_counter;

} Model_keypoint;

//
typedef struct struct_Scale_layer_flag {
  int exist[MAX_LAYER_NUMBER];

  struct_Scale_layer_flag() {
    for (int i = 0; i < MAX_LAYER_NUMBER; i++) exist[i] = 0;
  }
} Scale_layer_flag;

//
#define FEATRUE_BLOCK_WIDTH 0.04
//
namespace std {
template <>
struct hash<My_Type::Vector3i> {
  size_t operator()(const My_Type::Vector3i &vec3i) const {
    return ((((unsigned int)vec3i.x) * 73856093u) ^
            (((unsigned int)vec3i.y) * 19349669u) ^
            (((unsigned int)vec3i.z) * 83492791u));
  }
};
}  // namespace std

//!
/*!

*/
class Feature_map {
 public:
  //
  const int valid_observe_count = 5;
  const int outlyer_non_observe_count = valid_observe_count + 5;
  // Search radius for keypoint projection search
  const int search_projection_radius = 8;
  //
  float scale_layer_depth[NUMBER_OF_FEATURE_LEVEL];

  //
  bool feature_updated = true;

  // Map feature point mapper
  std::unordered_map<My_Type::Vector3i, int> map_point_mapper;
  // Model keypoint (contain valid informations)
  std::vector<Model_keypoint> model_keypoints;
  // Model features
  std::vector<cv::Mat> model_features;
  //
  std::vector<Scale_layer_flag> model_feature_scale_flag;

  //! Keyframe's feature mapper (from current frame index to fragment model
  //! index)
  std::vector<std::vector<int>> keyframe_feature_mapper_list;
  //! Data base
  std::vector<DBoW3::BowVector> dbow_vec_list;
  //! weight center of keyframe keypoints
  std::vector<My_Type::Vector3f> keyframe_weigth_centers;

  //
  Feature_map();
  ~Feature_map();

  //
  void update_current_features(std::vector<My_Type::Vector3f> current_keypoints,
                               cv::Mat current_features,
                               std::vector<int> &current_match_to_model,
                               std::vector<int> &previous_match_to_model,
                               My_pose &camera_pose);

  //
  void save_keyframe(cv::Mat current_features,
                     std::vector<int> current_match_to_model,
                     DBoW3::Vocabulary &feature_voc);

  // Searching for visible keypoints by projection culling
  void get_model_keypoints(
      std::vector<My_Type::Vector3f> &current_keypoints, My_pose &camera_pose,
      std::vector<std::vector<My_Type::Vector3f>> &visible_keypoints,
      std::vector<cv::Mat> &visible_features,
      std::vector<std::vector<int>> &visible_point_model_index);

  // Init feature map from last map
};
