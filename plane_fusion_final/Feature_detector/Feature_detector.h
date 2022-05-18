#pragma once

// OpenCV
#include <cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <helper_functions.h>
//
#include "Feature_detector/ORBextractor.h"
#include "OurLib/My_matrix.h"
#include "Preprocess_engine/Hierarchy_image.h"

//
#include "SLAM_system/SLAM_system_settings.h"

#define FEATURE_SCALE_FACTOR 1.3f
#define NUMBER_OF_FEATURE_LEVEL 8

class Feature_detector {
public:
  // ORB detector parameters
  const int aim_number_of_features = 300;
  const float orb_scale_factor = FEATURE_SCALE_FACTOR;
  const int number_of_feature_levels = NUMBER_OF_FEATURE_LEVEL;
  const int iniThFAST = 20;
  const int minThFAST = 7;
  //
  const int keypoint_window_radius = 10;

  //
  float scale_layer_depth[NUMBER_OF_FEATURE_LEVEL];

  // Current keypoints and features
  std::vector<cv::KeyPoint> current_keypoints;
  std::vector<My_Type::Vector3f> current_keypoint_position;
  cv::Mat current_features;

  // Visible model keypoints
  std::vector<std::vector<My_Type::Vector3f>> visible_model_keypoints;
  std::vector<cv::Mat> visible_model_features;
  std::vector<std::vector<int>> visible_point_model_index;

  //
  std::vector<cv::Point2f> current_keypoints_2d, previous_keypoints_2d;
  // Match keypoint index
  std::vector<int> current_match_to_model_id, previous_match_to_model_id;
  //
  int number_of_tracked_keypoints;

  // Gray image buffer
  cv::Mat gray_image, previous_gray_image, detected_mask;
  unsigned char *dev_gray_image_buffer;
  // ORB-SLAM2 ORB extractor
  ORB_SLAM2::ORBextractor *orb_feature_extractor;

  //!
  StopWatchInterface *timer_average;

  //
  Feature_detector();
  ~Feature_detector();
  // int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
  //
  void detect_orb_features(Hierarchy_image<float> &dev_current_intensity,
                           Hierarchy_image<My_Type::Vector3f> &current_points);

  //
  void match_orb_features(int number_of_model_keypoints);

private:
  //
  void prepare_to_detect_features();
};
