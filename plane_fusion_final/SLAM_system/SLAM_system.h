/**
 *  @file SLAM_system.h
 *  @brief Define 4 kinds of slam systems, which one has 7 base members.
 *  @details
 *  <pre> Base SLAM systems contains:
        1. Data_engine
        2. Preprocess_engine
        3. Track_engine
        4. Map_engine
        5. Plane_detector
        6. Associator, i.e. plane & keypoint
        7. Mesh_generator
    </pre>
 *  @endverbatim
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @todo Loop closure.
 */

#pragma once

#include <cstdio>
#include <functional>
#include <iostream>

#include "Associator/Associator.h"
#include "Data_engine/Data_engine.h"
#include "Feature_detector/Feature_detector.h"
#include "Map_engine/Map_engine.h"
#include "Map_engine/Map_optimizer.h"
#include "Map_engine/Mesh_generator.h"
#include "OurLib/My_pose.h"
#include "Plane_detector/Plane_detector.h"
#include "Preprocess_engine/Preprocess_engine.h"
#include "Track_engine/Track_engine.h"
using namespace std;

#include <cv.h>
#include <highgui.h>

#include <opencv2/opencv.hpp>

/**
 * @brief Enum SLAM system running state. stop, single frame, continuous frame.
 */
enum ProcessingState {
  /** STOP flag */
  STOP_PROCESS = 0,
  /** Running one frame */
  PROCESS_SINGLE_FRAME = 1,
  /** Running continuous frame */
  PROCESS_CONTINUOUS_FRAME = 2,
};

/**
 * @brief SLAM system base class.
 * @details
 */
class SLAM_system {
 public:
  /** @brief Flag whether generate mesh. */
  bool need_generate_mesh = false;

  /** @brief Frame id. */
  int frame_id = 0;

  /** @brief cv::Mat of color image. */
  cv::Mat color_mat;
  /** @brief cv::Mat of depth image. */
  cv::Mat depth_mat;

  /** @brief Timestamp of estimated camera pose. */
  double timestamp;
  /** @brief Estimated camera pose. */
  My_pose estimated_camera_pose;
  /** @brief Not sure. */
  My_pose pre_estimated_camera_pose;
  /** @brief Trajectory. */
  Trajectory estimated_trajectory;

  /** @brief Flag whether map update. */
  bool mesh_updated = true;
  /** @brief Flag whether finish all frames process. */
  bool all_data_process_done = false;
  /** @brief Flag whether optimize map. */
  bool map_need_optimization = false;

  /** @brief Data engine pointer. */
  Data_engine *data_engine;
  /** @brief Preprocess engine pointer. */
  Preprocess_engine *preprocess_engine;

  /** @brief Track engine pointer. */
  Track_engine *track_engine;
  /** @brief Map engine pointer. */
  Map_engine *map_engine;

  /** @brief Plane detector. */
  Plane_detector *plane_detector;
  /** @brief Feature detector. */
  Feature_detector *feature_detector;
  /** @brief Map optimization only for submap SLAM. */
  Associator *keypoint_associator;
  /** @brief Map optimization only for submap SLAM. */
  Associator *plane_associator;
  /** @brief Mesh generator for visualization. */
  Mesh_generator mesh_of_total_map;

  /** @brief SLAM timer. */
  StopWatchInterface *timer_average;

  /** @brief SLAM processing state. */
  ProcessingState processing_state = ProcessingState::STOP_PROCESS;
  /** @brief Process frame. */
  ProcessingState process_frames();

  // Relocalization
  // Loop closure

  /** @brief Default constructor.Initialize SLAM system settings. */
  SLAM_system();
  /** @brief Default destructor. */
  virtual ~SLAM_system(){};

  virtual void init(Data_engine *data_engine_ptr) = 0;

  virtual void generate_render_info(Eigen::Matrix4f view_pose) = 0;

  /** @brief End of process data. */
  virtual void end_of_process_data() = 0;

 protected:
  /** @brief Initialize SLAM system parameters. */
  void init_parameters(Data_engine *data_engine_ptr);

  /** @brief Create modules for different kinds of SLAM system. */
  virtual void init_modules() = 0;

  virtual void process_one_frame() = 0;

  virtual void preprocess() = 0;

  virtual void track_camera_pose() = 0;

  virtual void update_to_map() = 0;

  virtual void optimize_map() = 0;

  virtual void relocalization() = 0;

  virtual void detect_loop() = 0;

  virtual void generate_mesh() = 0;
};

/**
 * @brief Blank SLAM system. Only render basic OpenGL components for Debug.
 */
class Blank_SLAM_system : public SLAM_system {
 public:
  /** @brief Default constructor. */
  Blank_SLAM_system(){};
  /** @brief Default destructor. */
  ~Blank_SLAM_system(){};

  void init();

 protected:
  void process_one_frame() override;

 private:
  // remove warning: hide overloaded virtual function.
  // reference:
  // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
  using SLAM_system::init;
};

/**
 * @brief Directly using the ground_truth.
 */
class Ground_truth_SLAM_system : public SLAM_system {
 public:
  /** @brief Default constructor. */
  Ground_truth_SLAM_system();
  /** @brief Default destructor. */
  ~Ground_truth_SLAM_system();

  void init(Data_engine *data_engine_ptr) override;

  void generate_render_info(Eigen::Matrix4f view_pose) override;

  /** @brief End of process data. */
  void end_of_process_data() override;

 protected:
  /** @brief Create modules for different kinds of SLAM system. */
  void init_modules() override;

  void process_one_frame() override;

  void preprocess() override;

  void track_camera_pose() override;

  void update_to_map() override;

  void optimize_map() override{};

  void relocalization() override{};

  void detect_loop() override{};

  void generate_mesh() override;
};

/**
 * @brief Basic voxel-based SLAM system
 */
class Basic_voxel_SLAM_system : public SLAM_system {
 public:
  /** @brief Default constructor. */
  Basic_voxel_SLAM_system();
  /** @brief Default destructor. */
  ~Basic_voxel_SLAM_system();

  void init(Data_engine *data_engine_ptr) override;

  void generate_render_info(Eigen::Matrix4f view_pose) override;

  /** @brief End of process data. */
  void end_of_process_data() override;

 protected:
  /** @brief Create modules for different kinds of SLAM system. */
  void init_modules() override;

  void process_one_frame() override;

  void preprocess() override;

  void track_camera_pose() override;

  void update_to_map() override;

  void optimize_map() override{};

  void relocalization() override{};

  void detect_loop() override{};

  void generate_mesh() override;
};

/**
 * @brief Submap_SLAM_system
 */
class Submap_SLAM_system : public SLAM_system {
 public:
  /** @brief Enable inter-submap optimization module. */
  bool enable_optimization = true;
  /** @brief Enable loop detection module. */
  bool enable_loop_detection = true;
  /** @brief Enable submap module. */
  bool enable_submap_creation = false;
  /** @brief Enable plane map module. */
  bool enable_plane_map = true;

  /** @brief Voxel map of each submap. */
  std::vector<Submap_Voxel_map *> submap_ptr_array;
  /** @brief Feature map of each submap. */
  std::vector<Feature_map *> feature_map_ptr_array;

  /** @brief Triangle mesh of each submap. */
  std::vector<Mesh_generator *> mesh_ptr_array;
  /** @brief Estiamted trajectory of each submap. */
  std::vector<Trajectory> estimated_trajectory_array;
  /** @brief Submap pose matrices. */
  std::vector<My_pose *> submap_pose_array;

  /** @brief Vocabulary data structure. */
  DBoW3::Vocabulary ORB_vocabulary;

  /** @brief Default constructor. */
  Submap_SLAM_system();
  /** @brief Default destructor. */
  ~Submap_SLAM_system();

  /** @brief Debug. */
  std::vector<Eigen::Vector3f> keypoint_buffer_1, keypoint_buffer_2;

  void init(Data_engine *data_engine_ptr) override;

  void generate_render_info(Eigen::Matrix4f view_pose) override;

  /** @brief End of process data. */
  void end_of_process_data() override;

 protected:
  /** @brief Create modules for different kinds of SLAM system. */
  void init_modules() override;

  void process_one_frame() override;

  void preprocess() override;

  void track_camera_pose() override;

  void update_to_map() override;

  void optimize_map() override;

  void relocalization() override{};

  void detect_loop() override;

  void generate_mesh() override;

 private:
  void generate_submap_to_global_plane_mapper(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container);

  void generate_submap_to_global_plane_relabel_list(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container,
      int submap_id, std::vector<My_Type::Vector2i> &relabel_list);

  void filter_loop_matches(std::vector<Eigen::Vector3f> &current_points,
                           std::vector<Eigen::Vector3f> &loop_points,
                           std::vector<Plane_info> &current_planes,
                           std::vector<Plane_info> &loop_planes,
                           std::vector<bool> &is_valid_keypoint_match,
                           std::vector<bool> &is_valid_plane_match);

  void match_plane_by_parameter(
      std::vector<Plane_info> &previous_map_planes,
      std::vector<Plane_info> &current_map_planes,
      std::vector<std::pair<int, int>> &plane_matches);
};

/**
 * @brief Semantic mapping SLAM system.
 * @details Designed for ScanNet dataset mode, take ground truth as camera
 *          poses.
 */
class Somapping_SLAM_system : public Submap_SLAM_system {
 public:
  /** @brief Enable inter-submap optimization module. */
  bool enable_optimization = true;
  /** @brief Enable loop detection module. */
  bool enable_loop_detection = true;
  /** @brief Enable submap module. */
  bool enable_submap_creation = false;
  /** @brief Enable plane map module. */
  bool enable_plane_map = true;

  /** @brief Voxel map of each submap. */
  std::vector<Submap_Voxel_map *> submap_ptr_array;
  /** @brief Feature map of each submap. */
  std::vector<Feature_map *> feature_map_ptr_array;

  /** @brief Triangle mesh of each submap. */
  std::vector<Mesh_generator *> mesh_ptr_array;
  /** @brief Estiamted trajectory of each submap. */
  std::vector<Trajectory> estimated_trajectory_array;
  /** @brief Submap pose matrices. */
  std::vector<My_pose *> submap_pose_array;

  /** @brief Vocabulary data structure. */
  DBoW3::Vocabulary ORB_vocabulary;

  /** @brief Default constructor. */
  Somapping_SLAM_system();
  /** @brief Default destructor. */
  ~Somapping_SLAM_system();

  /** @brief Debug. */
  std::vector<Eigen::Vector3f> keypoint_buffer_1, keypoint_buffer_2;

  void init(Data_engine *data_engine_ptr) override;

  void generate_render_info(Eigen::Matrix4f view_pose) override;

  /** @brief End of process data. */
  void end_of_process_data() override;

  protected:
  /** @brief Create modules for different kinds of SLAM system. */
  void init_modules() override;

  void process_one_frame() override;

  void preprocess() override;

  void track_camera_pose() override;

  void update_to_map() override;

  void optimize_map() override;

  void relocalization() override{};

  void detect_loop() override;

  void generate_mesh() override;

  private:
  void generate_submap_to_global_plane_mapper(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container);

  void generate_submap_to_global_plane_relabel_list(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container,
      int submap_id, std::vector<My_Type::Vector2i> &relabel_list);

  void filter_loop_matches(std::vector<Eigen::Vector3f> &current_points,
                           std::vector<Eigen::Vector3f> &loop_points,
                           std::vector<Plane_info> &current_planes,
                           std::vector<Plane_info> &loop_planes,
                           std::vector<bool> &is_valid_keypoint_match,
                           std::vector<bool> &is_valid_plane_match);

  void match_plane_by_parameter(
      std::vector<Plane_info> &previous_map_planes,
      std::vector<Plane_info> &current_map_planes,
      std::vector<std::pair<int, int>> &plane_matches);
};
