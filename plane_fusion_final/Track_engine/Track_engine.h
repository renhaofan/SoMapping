#pragma once

//
#include <iostream>
#include <vector>
// Eigen
#include <Eigen/Dense>

// My pose
//#include "OurLib/My_pose.h"
#include "Preprocess_engine/Hierarchy_image.h"
#include "Track_structure.h"
//
#include "SLAM_system/SLAM_system_settings.h"

//! Tracking state
/*!

*/
enum TrackingState {
  BEFORE_TRACKING,
  DURING_TRACKING,
  TRACKING_FAILED,
  TRACKINE_SUCCED,
};

//!
/*!

*/
class Track_engine {
public:
#pragma region(Residual variables)

  //! Total hessian matrix
  Eigen::Matrix<float, 6, 6> total_hessian;
  //! Total nabla matrix
  Eigen::Matrix<float, 6, 1> total_nabla;
  //!
  // const float icp_huber; /* set as sensor noise radius */
  const float keypoint_huber = 0.04;

#pragma endregion

#pragma region(Tracking state)
  //!
  TrackingState tracking_state = TrackingState::BEFORE_TRACKING;

  //! The incremental pose
  Eigen::Matrix4f incremental_pose;
#pragma endregion

#ifdef COMPILE_DEBUG_CODE
  //!
  My_Type::Vector3f *correspondence_lines, *dev_correspondence_lines;
#endif

  //! Default constructor/destructor
  Track_engine();
  ~Track_engine();

  //! Get tracking state
  /*!

      */
  TrackingState get_track_state() { return this->tracking_state; }

  //! Init module
  /*!

      */
  virtual void init();

  //! Generate correspondence lines
  /*!

      */
#ifdef COMPILE_DEBUG_CODE
  virtual void generate_icp_correspondence_lines(
      const Hierarchy_image<My_Type::Vector3f> &dev_current_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_points_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_current_normals_hierarchy,
      const Hierarchy_image<My_Type::Vector3f> &dev_model_normals_hierarchy);
#endif

protected:
  //! Prepare to track new frame
  /*!
              \note	Reset all ICP variable and tracking state.
      */
  virtual void prepare_to_track_new_frame() = 0;

  //! Prepare to run new iteration
  /*!

      */
  virtual void prepare_to_new_iteration() = 0;

  //! Update camera pose
  /*!

      */
  void update_camera_pose(Eigen::Matrix4f &camera_pose) const;
};

//! Basic ICP tracker. Only using depth information estimate camera pose.
/*!


*/
class Basic_ICP_tracker : public Track_engine {
public:
  //! Increment pose matrix
  My_Type::Matrix44f cache_increment_pose;

  //! Point-Plane hessian,nabla matrix
  Accumulate_result icp_accumulation, *dev_accumulation_buffer;

  //! Default constructor/destructor
  Basic_ICP_tracker();
  ~Basic_ICP_tracker();

  //! Track camera pose
  /*!

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
  //! Prepare to track new frame
  /*!
              \note	Reset all ICP variable and tracking state.
      */
  void prepare_to_track_new_frame() override;

  //! Prepare to run new iteration
  /*!

      */
  void prepare_to_new_iteration() override;
};

//
class Keypoint_ICP_tracker : public Track_engine {
public:
  //! Increment pose matrix
  My_Type::Matrix44f cache_increment_pose;

  //! Point-Plane hessian,nabla matrix
  Accumulate_result icp_accumulation, photometric_accumulation,
      *dev_accumulation_buffer;

  //! Default constructor/destructor
  Keypoint_ICP_tracker();
  ~Keypoint_ICP_tracker();

  //! Track camera pose
  /*!

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
  //! Prepare to track new frame
  /*!
      \note	Reset all ICP variable and tracking state.
      */
  void prepare_to_track_new_frame() override;

  //! Prepare to run new iteration
  /*!

      */
  void prepare_to_new_iteration() override;
};
