/**
 *  @file Track_engine.h
 *  @brief Track to estimate camera pose
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

// My pose
//#include "OurLib/My_pose.h"
#include "Preprocess_engine/Hierarchy_image.h"
#include "SLAM_system/SLAM_system_settings.h"
#include "Track_structure.h"

/**
 * @brief TrackingState enum
 */
enum TrackingState {
  /** Before tracing */
  BEFORE_TRACKING,
  /** During tracing */
  DURING_TRACKING,
  /** Faild to track */
  TRACKING_FAILED,
  /** Track successfully */
  TRACKINE_SUCCED,
};

class Track_engine {
 public:
#if __unix__
#pragma region "Residual variables" {
#elif _WIN32
#pragma region(Residual variables)
#endif
  /**
   * @brief Total hessian matrix
   */
  Eigen::Matrix<float, 6, 6> total_hessian;
  /**
   * @brief Total nabla matrix
   */
  Eigen::Matrix<float, 6, 1> total_nabla;

  // const float icp_huber; /* set as sensor noise radius */
  const float keypoint_huber = 0.04;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Tracking state" {
#elif _WIN32
#pragma region(Tracking state)
#endif
  TrackingState tracking_state = TrackingState::BEFORE_TRACKING;
  /**
   * @brief incremental_pose.
   */
  Eigen::Matrix4f incremental_pose;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#ifdef COMPILE_DEBUG_CODE
  My_Type::Vector3f *correspondence_lines, *dev_correspondence_lines;
#endif

  /**
   * @brief Default constructor.
   */
  Track_engine();
  /**
   * @brief Default destructor.
   */
  virtual ~Track_engine() = 0;

  /**
   * @brief Get tracking state.
   * @return Tracking state.
   */
  TrackingState get_track_state() { return this->tracking_state; }

  /**
   * @brief Init module.
   */
  virtual void init();

  //! Generate correspondence lines
#ifdef COMPILE_DEBUG_CODE
  virtual void generate_icp_correspondence_lines(
      const Hierarchy_image<My_Type::Vector3f> &dev_current_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_current_normals_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_normals_hierarchy);
#endif

 protected:
  /**
   * @brief Prepare to track new frame.
   * @note Reset all ICP variable and tracking state.
   */
  virtual void prepare_to_track_new_frame() = 0;

  /**
   * @brief Prepare to run new iteration
   */
  virtual void prepare_to_new_iteration() = 0;

  /**
   * @brief Update camera pose.
   * @param camera_pose Matrix representation.
   */
  void update_camera_pose(Eigen::Matrix4f &camera_pose) const;
};

/**
 * @brief Basic ICP tracker. Only using depth information estimate camera pose.
 */
class Basic_ICP_tracker : public Track_engine {
 public:
  /**
   * @brief Increment pose matrix.
   */
  My_Type::Matrix44f cache_increment_pose;

  /**
   * @brief Point-Plane hessian,nabla matrix
   */
  Accumulate_result icp_accumulation;
  /**
   * @brief Point-Plane hessian,nabla matrix
   */
  Accumulate_result *dev_accumulation_buffer;

  /**
   * @brief Default constructor
   */
  Basic_ICP_tracker();
  /**
   * @brief Default destructor
   */
  ~Basic_ICP_tracker();

  /**
   * @brief Track camera pose
   * @param dev_current_points_hierarchy
   * @param dev_model_points_hierarchy
   * @param dev_current_normals_hierarchy
   * @param dev_model_normals_hierarchy
   * @param dev_current_intensity
   * @param dev_model_intensity
   * @param dev_model_gradient
   * @param camera_pose
   * @return
   */
  TrackingState track_camera_pose(
      const Hierarchy_image<My_Type::Vector3f> &dev_current_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_current_normals_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_normals_hierarchy,
      const Hierarchy_image<float> &dev_current_intensity,
      const Hierarchy_image<float> &dev_model_intensity,
      const Hierarchy_image<My_Type::Vector2f> &dev_model_gradient,
      Eigen::Matrix4f &camera_pose);

 protected:
  /**
   * @brief Prepare to track new frame.
   * @note Reset all ICP variable and tracking state.
   */
  void prepare_to_track_new_frame() override;
  /**
   * @brief Prepare to run new iteration
   */
  void prepare_to_new_iteration() override;
};

class Keypoint_ICP_tracker : public Track_engine {
 public:
  /**
   * @brief Increment pose matrix.
   */
  My_Type::Matrix44f cache_increment_pose;

  /**
   * @brief Point-Plane hessian,nabla matrix.
   */
  Accumulate_result icp_accumulation;
  /**
   * @brief Point-Plane hessian,nabla matrix.
   */
  Accumulate_result photometric_accumulation;
  /**
   * @brief Point-Plane hessian,nabla matrix.
   */
  Accumulate_result *dev_accumulation_buffer;

  /**
   * @brief Default constructor
   */
  Keypoint_ICP_tracker();
  /**
   * @brief Default destructor
   */
  ~Keypoint_ICP_tracker();

  /**
   * @brief Track camera pose
   * @param dev_current_points_hierarchy
   * @param dev_model_points_hierarchy
   * @param dev_current_normals_hierarchy
   * @param dev_model_normals_hierarchy
   * @param dev_current_intensity
   * @param dev_model_intensity
   * @param dev_model_gradient
   * @param current_keypoints
   * @param model_keypoints
   * @param camera_pose
   * @return
   */
  TrackingState track_camera_pose(
      const Hierarchy_image<My_Type::Vector3f> &dev_current_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_current_normals_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_normals_hierarchy,
      const Hierarchy_image<float> &dev_current_intensity,
      const Hierarchy_image<float> &dev_model_intensity,
      const Hierarchy_image<My_Type::Vector2f> &dev_model_gradient,
      const std::vector<Eigen::Vector3f> &current_keypoints,
      const std::vector<Eigen::Vector3f> &model_keypoints,
      Eigen::Matrix4f &camera_pose);

 protected:
  /**
   * @brief Prepare to track new frame.
   * @note Reset all ICP variable and tracking state.
   */
  void prepare_to_track_new_frame() override;

  /**
   * @brief Prepare to run new iteration
   */
  void prepare_to_new_iteration() override;
};
