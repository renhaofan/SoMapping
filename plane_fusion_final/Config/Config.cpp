#include "Config.h"

JSON_CONFIG* JSON_CONFIG::instance_ptr = nullptr;

void JSON_CONFIG::init() {
  //  std::ifstream jfile("../../plane_fusion_final/Config/config.json");
  //  if (!jfile.is_open()) {
  //#ifdef LOGGING
  //    LOG_WARNING("Failed to open json files, using default value.");
  //#endif
  //  }
  //  jfile >> j;
  //  jfile.close();
  //  if (jfile.is_open()) {
  //#ifdef LOGGING
  //    LOG_ERROR("Failed to close json files.");
  //#endif
  //}
  j = json(R"({
  "PlaneDetection" : {
    "MIN_PLANE_DISTANCE" : 0.02,
    "HISTOGRAM_STEP" : 0.05,
    "HISTOGRAM_WIDTH" : 128,
    "MAX_CURRENT_PLANES" : 64,
    "MAX_MODEL_PLANES" : 1024
  },
  "SLAM" : {
    "Blank_SLAM_system" : false,
    "Ground_truth_SLAM_system" : false,
    "Basic_voxel_SLAM_system" : false,
    "Submap_SLAM_system" : false,
    "Somapping_SLAM_system" : true
  },
  "scene" : {
    "name" : "scene0000_00",
    "path" : "/home/steve/dataset/scene0000_00"
  }
})"_json);
}

JSON_CONFIG::JSON_CONFIG() { init(); }

JSON_CONFIG::~JSON_CONFIG() {}
