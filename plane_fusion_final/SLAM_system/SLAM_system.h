#pragma once

//
#include "Associator/Associator.h"
#include "Data_engine/Data_engine.h"
#include "Feature_detector/Feature_detector.h"
#include "Map_engine/Map_engine.h"
#include "Map_engine/Map_optimizer.h"
#include "Map_engine/Mesh_generator.h"
#include "Plane_detector/Plane_detector.h"
#include "Preprocess_engine/Preprocess_engine.h"
#include "Track_engine/Track_engine.h"

//
#include "OurLib/My_pose.h"

// std io
#include <cstdio>
#include <functional>
#include <iostream>
using namespace std;

// OpenCV
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>

//! System run state
/*!

*/
enum ProcessingState {
  STOP_PROCESS = 0,
  PROCESS_SINGLE_FRAME = 1,
  PROCESS_CONTINUOUS_FRAME = 2,
};

//!
/*!

        \note	This is a base class (working as an interface)
*/
class SLAM_system {
public:
  //!
  bool need_generate_mesh = false;
  //!
  int frame_id = 0;

  //! cv::Mat of color image.
  cv::Mat color_mat;
  //! cv::Mat of depth image.
  cv::Mat depth_mat;

  //! Estimated camera pose
  My_pose estimated_camera_pose;
  My_pose pre_estimated_camera_pose;
  //! Timestamp of estimated camera pose
  double timestamp;

  //! Flag for map update
  bool mesh_updated = true;
  //! Finished process
  bool all_data_process_done = false;
  //! Flag for optimization
  bool map_need_optimization = false;

  //! Data engine pointer
  Data_engine *data_engine;
  //! Preprocess engine pointer
  Preprocess_engine *preprocess_engine;
  //! Plane detector
  Plane_detector *plane_detector;
  //
  Feature_detector *feature_detector;
  //! Track engine pointer
  Track_engine *track_engine;
  //! Map engine pointer
  Map_engine *map_engine;
  //! Map optimization /* for submap SLAM only */
  Associator *keypoint_associator;
  Associator *plane_associator;
  //! Relocalization

  //! Loop closure

  //! Mesh generator for visualization
  Mesh_generator mesh_of_total_map;
  //! Trajectory
  Trajectory estimated_trajectory;

  //!
  StopWatchInterface *timer_average;

  //! SLAM processing state
  ProcessingState processing_state = ProcessingState::STOP_PROCESS;

  //!
  ProcessingState process_frames();

  //!
  SLAM_system();
  virtual ~SLAM_system(){};

  //!
  virtual void init(Data_engine *data_engine_ptr) = 0;

  //!
  virtual void generate_render_info(Eigen::Matrix4f view_pose) = 0;

  //! End of process data
  virtual void end_of_process_data() = 0;

protected:
  //! Initialize SLAM system parameters
  void init_parameters(Data_engine *data_engine_ptr);

  //! Create modules for different kinds of SLAM system
  virtual void init_modules() = 0;

  //!
  virtual void process_one_frame() = 0;

  //!
  virtual void preprocess() = 0;

  //!
  virtual void track_camera_pose() = 0;

  //!
  virtual void update_to_map() = 0;

  //!
  virtual void optimize_map() = 0;

  //!
  virtual void relocalization() = 0;

  //!
  virtual void detect_loop() = 0;

  //!
  virtual void generate_mesh() = 0;
};

//! Blank SLAM system. Only render basic OpenGL components for Debug.
/*!

*/
class Blank_SLAM_system : public SLAM_system {
public:
  //! Default constructor/destructor
  Blank_SLAM_system(){};
  ~Blank_SLAM_system(){};

  //!
  void init();

protected:
  //!
  void process_one_frame() override;
};

//! Directly using the ground_truth for reconstruction
/*!


*/
class Ground_truth_SLAM_system : public SLAM_system {
public:
  //! Default constructor/destructor
  Ground_truth_SLAM_system();
  ~Ground_truth_SLAM_system();

  //!
  void init(Data_engine *data_engine_ptr) override;

  //!
  void generate_render_info(Eigen::Matrix4f view_pose) override;

  //! End of process data
  void end_of_process_data() override;

protected:
  //! Create modules for different kinds of SLAM system
  void init_modules() override;

  //!
  void process_one_frame() override;

  //!
  void preprocess() override;

  //!
  void track_camera_pose() override;

  //!
  void update_to_map() override;

  //!
  void optimize_map() override{};

  //!
  void relocalization() override{};

  //!
  void detect_loop() override{};

  //!
  void generate_mesh() override;
};

//! Basic voxel-based SLAM system
/*!



*/
class Basic_voxel_SLAM_system : public SLAM_system {
public:
  //! Default constructor/destructor
  Basic_voxel_SLAM_system();
  ~Basic_voxel_SLAM_system();

  //!
  void init(Data_engine *data_engine_ptr) override;

  //!
  void generate_render_info(Eigen::Matrix4f view_pose) override;

  //! End of process data
  void end_of_process_data() override;

protected:
  //! Create modules for different kinds of SLAM system
  void init_modules() override;

  //!
  void process_one_frame() override;

  //!
  void preprocess() override;

  //!
  void track_camera_pose() override;

  //!
  void update_to_map() override;

  //!
  void optimize_map() override{};

  //!
  void relocalization() override{};

  //!
  void detect_loop() override{};

  //!
  void generate_mesh() override;
};

//! Submap_SLAM_system
/*!

*/
class Submap_SLAM_system : public SLAM_system {
public:
  //! Enable inter-submap optimization module
  bool enable_optimization = true;
  //! Enable loop detection module
  bool enable_loop_detection = true;
  //! Enable submap module
  bool enable_submap_creation = false;
  //! Enable plane map module
  bool enable_plane_map = true;

  //! Voxel map of each submap
  std::vector<Submap_Voxel_map *> submap_ptr_array;
  //! Feature map of each submap
  std::vector<Feature_map *> feature_map_ptr_array;
  //! Triangle mesh of each submap
  std::vector<Mesh_generator *> mesh_ptr_array;
  //! Estiamted trajectory of each submap
  std::vector<Trajectory> estimated_trajectory_array;
  //! Submap pose matrixes
  std::vector<My_pose *> submap_pose_array;

  //! Vocabulary data structure
  DBoW3::Vocabulary ORB_vocabulary;

  //! Default constructor/destructor
  Submap_SLAM_system();
  ~Submap_SLAM_system();

  //! Debug
  std::vector<Eigen::Vector3f> keypoint_buffer_1, keypoint_buffer_2;

  //!
  void init(Data_engine *data_engine_ptr) override;

  //!
  void generate_render_info(Eigen::Matrix4f view_pose) override;

  //! End of process data
  void end_of_process_data() override;

protected:
  //! Create modules for different kinds of SLAM system
  void init_modules() override;

  //!
  void process_one_frame() override;

  //!
  void preprocess() override;

  //!
  void track_camera_pose() override;

  //!
  void update_to_map() override;

  //!
  void optimize_map() override;

  //!
  void relocalization() override{};

  //!
  void detect_loop() override;

  //!
  void generate_mesh() override;

private:
  //!
  void generate_submap_to_global_plane_mapper(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container);

  //!
  void generate_submap_to_global_plane_relabel_list(
      std::vector<std::vector<My_Type::Vector2i>> &global_plane_container,
      int submap_id, std::vector<My_Type::Vector2i> &relabel_list);

  //!
  void filter_loop_matches(std::vector<Eigen::Vector3f> &current_points,
                           std::vector<Eigen::Vector3f> &loop_points,
                           std::vector<Plane_info> &current_planes,
                           std::vector<Plane_info> &loop_planes,
                           std::vector<bool> &is_valid_keypoint_match,
                           std::vector<bool> &is_valid_plane_match);

  //!
  void
  match_plane_by_parameter(std::vector<Plane_info> &previous_map_planes,
                           std::vector<Plane_info> &current_map_planes,
                           std::vector<std::pair<int, int>> &plane_matches);
};
