#pragma once

// C/C++ IO
#include <cstdio>
#include <iostream>
using namespace std;

// Our lib
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

//! Data loader
/*!

*/
class Data_loader {
public:
  //! Default constructor/destructor
  Data_loader();
  ~Data_loader();

  //! Read next frame
  /*!
          \param	timestamp		timestamp of captured frame

          \param	color_mat		Color image matrix. (cv::Mat)

          \param	depth_mat		Depth image matrix. (cv::Mat)

          \param	show_in_opencv	Show images in OpenCV windows.

          \return	bool	true	-	succeed	load images
                                          false	-	failed to load images
  */
  bool load_next_frame(double &timestamp, cv::Mat &color_mat,
                       cv::Mat &depth_mat, bool show_in_opencv = false);

  //! Jump to begin

  //!
  /*!
          \param	ground_truth_path	Path of ground truth file.

          \return void
  */
  void load_ground_truth(string ground_truth_path,
                         bool is_ICL_NUIM_data = false);

  //! Get Image_loader
  Image_loader *get_image_loader() const { return this->image_loader; }

  //!
  void get_depth_image_size(int &width, int &height) const {
    this->image_loader->get_depth_image_size(width, height);
  }

  //!
  void get_color_image_size(int &width, int &height) const {
    this->image_loader->get_color_image_size(width, height);
  }

  //!
  size_t ground_truth_camera_pose_index = 0;
  //! Get next ground_truth camera pose
  bool get_next_ground_truth_camera_pose(Trajectory_node &trajectory_node);

  //! Get ground_truth trajectory
  const Trajectory &get_ground_truth_trajectory() const {
    return this->ground_truth_trajectory;
  }
  //! Get comparison trajectory
  const Trajectory &get_comparison_trajectory() const {
    return this->comparison_trajectory;
  }
  //! Get OpenGL FreeView camera trajectory trajectory
  const Trajectory &get_OpenGL_camera_freeview_trajectory() const {
    return this->GLcamera_freeview_trajectory;
  }
  //! Get OpenGL key view (TopView, etc) camera pose trajectory
  const Trajectory &get_OpenGL_camera_keyview_trajectory() const {
    return this->GLcamera_keyview_trajectory;
  }

protected:
  //! Image loader
  Image_loader *image_loader;

#pragma region(Information of trajectories)

  //! Whether ground_truth trajectory file exist.
  bool with_ground_truth_trajectory = false;
  //! Whether comparison trajectory file exist.
  bool with_comparison_trajectory = false;
  //! Whether OpenGL FreeView camera trajectory file exist.
  bool with_GLcamera_freeview_trajectory = false;
  //! Whether OpenGL key view (TopView, etc) camera pose file exist.
  bool with_GLcamera_keyview_pose = false;

  //! ground_truth trajectory file path.
  string ground_truth_trajectory_path;
  //! Comparison trajectory file path.
  string comparison_trajectory_path;
  //! OpenGL FreeView camera trajectory file path.
  string GLcamera_freeview_trajectory_path;
  //! OpenGL key view (TopView, etc) camera pose file path.
  string GLcamera_keyview_pose_path;

  // Trajectories
  //! Ground truth trajectory
  Trajectory ground_truth_trajectory;
  //! Comparison trajectory
  Trajectory comparison_trajectory;
  //! OpenGL camera FreeView trajectory
  Trajectory GLcamera_freeview_trajectory;
  //! OpenGL camera key view (TopView, etc) trajectory
  Trajectory GLcamera_keyview_trajectory;

#pragma endregion

  //! Initialize Image_loader
  void init_image_loader(Image_loader *image_loader_ptr);

private:
  //! Show current frame
  /*!
          \param	color_mat	cv::Mat of color image.

          \param	depth_mat	cv::Mat of depth image.

          \return void
  */
  void show_current_frame(cv::Mat &color_mat, cv::Mat &depth_mat);
};
