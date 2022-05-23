/**
 *  Copyright (C) All rights reserved.
 *  @file Data_loader.h
 *  @brief Load frame by member image_loader. And load camera pose ground truth.
 *  In addition, some GLCamera trajectory path(not used tmp).
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include <cstdio>
#include <iostream>
using namespace std;

#include "Image_loader.h"
#include "OurLib/Trajectory_node.h"

//// DBoW3
//#include "DBoW3.h"
////
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//// check weather conrib exist
//#ifdef USE_CONTRIB
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/xfeatures2d.hpp>
////#include <opencv2/cudafeatures2d.hpp>
//#endif
//#include "DescManip.h"

/**
 * @brief Load frame and ground truth of camera pose.
 */
class Data_loader {
 public:
  /** @brief Default constructor. */
  Data_loader();
  /** @brief Default deconstructor. */
  ~Data_loader();

  /**
   * @brief Read color and depth frame by call function
   *        this->image_loader->load_next_frame();
   * @param timestamp Timestamp of captured frame.
   * @param color_mat Color image matrix. (cv::Mat)
   * @param depth_mat Depth image matrix. (cv::Mat)
   * @param show_in_opencv Call function cv::imshow(color_mat) for DBUG, disable
   *        by default.
   * @exception this->image_loader == nullptr
   * @return true/false, succeed/failed to load images.
   */
  bool load_next_frame(double &timestamp, cv::Mat &color_mat,
                       cv::Mat &depth_mat, bool show_in_opencv = false);
  /**
   * @brief Load camera pose ground truth.
   * @param ground_truth_path Path of ground truth file.
   * @param is_ICL_NUIM_data, whose GT is based on left-hand, false by default.
   */
  void load_ground_truth(string ground_truth_path,
                         bool is_ICL_NUIM_data = false);

  /**
   * @brief Get the this->image_loader pointer.
   * @return (Image_loader*), i.e. this->image_loader pointer.
   */
  Image_loader *get_image_loader() const { return this->image_loader; }
  /**
   * @brief Extract depth width and height by two variables by calling
   *        Image_loader function.
   * @param width Depth image width.
   * @param height Depth image width.
   */
  void get_depth_image_size(int &width, int &height) const {
    this->image_loader->get_depth_image_size(width, height);
  }
  /**
   * @brief Extract color width and height by two variables by calling
   *        Image_loader function.
   * @param width Color image width.
   * @param height Color image width.
   */
  void get_color_image_size(int &width, int &height) const {
    this->image_loader->get_color_image_size(width, height);
  }

  /**
   * @brief Camera pose index of ground truth.
   */
  size_t ground_truth_camera_pose_index = 0;
  /**
   * @brief Get cmaera pose from ground truth.
   * @param trajectory_node Contains translation, rotation in specified
   * timestamp.
   * @return true/false, success/failure.
   */
  bool get_next_ground_truth_camera_pose(Trajectory_node &trajectory_node);

  /**
   * @brief Get ground_truth trajectory.
   * @return const Trajectory reference.
   */
  const Trajectory &get_ground_truth_trajectory() const {
    return this->ground_truth_trajectory;
  }
  /**
   * @brief Get comparison trajectory.
   * @return const Trajectory reference.
   */
  const Trajectory &get_comparison_trajectory() const {
    return this->comparison_trajectory;
  }

  /**
   * @brief Get OpenGL FreeView camera trajectory trajectory.
   * @return const Trajectory reference.
   */
  const Trajectory &get_OpenGL_camera_freeview_trajectory() const {
    return this->GLcamera_freeview_trajectory;
  }
  /**
   * @brief Get OpenGL key view (TopView, etc) camera pose trajectory
   * @return const Trajectory reference.
   */
  const Trajectory &get_OpenGL_camera_keyview_trajectory() const {
    return this->GLcamera_keyview_trajectory;
  }

 protected:
  /** @brief Image loader pointer. */
  Image_loader *image_loader;

  /** @brief Whether ground_truth trajectory file exist. */
  bool with_ground_truth_trajectory = false;
  /** @brief Whether comparison trajectory file exist. */
  bool with_comparison_trajectory = false;
  /** @brief Whether OpenGL FreeView camera trajectory file exist. */
  bool with_GLcamera_freeview_trajectory = false;
  /** @brief Whether OpenGL key view (TopView, etc) camera pose file exist. */
  bool with_GLcamera_keyview_pose = false;

  /** @brief ground_truth trajectory file path. */
  string ground_truth_trajectory_path;
  /** @brief Comparison trajectory file path. */
  string comparison_trajectory_path;
  /** @brief OpenGL FreeView camera trajectory file path. */
  string GLcamera_freeview_trajectory_path;
  /** @brief OpenGL key view (TopView, etc) camera pose file path. */
  string GLcamera_keyview_pose_path;

  /** @brief Ground truth trajectory. */
  Trajectory ground_truth_trajectory;
  /** @brief Comparison trajectory. */
  Trajectory comparison_trajectory;
  /** @brief OpenGL camera FreeView trajectory. */
  Trajectory GLcamera_freeview_trajectory;
  /** @brief OpenGL camera key view (TopView, etc) trajectory. */
  Trajectory GLcamera_keyview_trajectory;

  /**
   * @brief Initialize the this image_loader.
   * @param image_loader_ptr Assigned pointer for this->image_loader.
   * @warning If param image_loader_ptr is equal to nullptr, this->image_loader
   * will be Blank_image_loader().
   */
  void init_image_loader(Image_loader *image_loader_ptr);

 private:
  /**
   * @brief Show current color and depth frame for DBUG.
   * @param color_mat cv::Mat of color image.
   * @param depth_mat cv::Mat of depth image.
   */
  void show_current_frame(cv::Mat &color_mat, cv::Mat &depth_mat);
};
