/**
 *  @file SLAM_system_settings.cpp
 *  @brief SLAM system settings
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#include "SLAM_system_settings.h"
#include "Log.h"

#include "OurLib/my_math_functions.h"

SLAM_system_settings *SLAM_system_settings::instance_ptr = nullptr;

SLAM_system_settings::SLAM_system_settings() { this->set_to_default(); }
SLAM_system_settings::~SLAM_system_settings() {}

void SLAM_system_settings::set_to_default() {
  this->generate_mesh_for_visualization = true;
  this->enable_plane_module = true;

  this->image_alginment_patch_width = 16;
  this->raycast_range_patch_width = 16;

  this->aligned_depth_size.width = 0;
  this->aligned_depth_size.height = 0;
  this->aligned_color_size.width = 0;
  this->aligned_color_size.height = 0;

  /*
   Note that the depth images provided in our dataset are already pre-registered
   to the RGB images. Therefore, rectifying the depth images based on the
   intrinsic parameters is not straight forward.
   https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
   */
  //   TUM 16-bit PNG files, see web above.

  //  this->sensor_params.sensor_scale = 5000.0f;
  //  int calib_mode = 5;
  //  if (calib_mode == 0) {
  //    // ICL
  //    this->sensor_params.sensor_fx = 480.0f;
  //    this->sensor_params.sensor_fy = 480.0f;
  //    this->sensor_params.sensor_cx = 320.0f;
  //    this->sensor_params.sensor_cy = 240.0f;
  //  } else if (calib_mode == 1) {
  //    // TUM Fr1
  //    this->sensor_params.sensor_fx = 591.1;
  //    this->sensor_params.sensor_fy = 590.1;
  //    this->sensor_params.sensor_cx = 331.0;
  //    this->sensor_params.sensor_cy = 234.0;
  //  } else if (calib_mode == 2) {
  //    // TUM Fr2
  //    this->sensor_params.sensor_fx = 580.8;
  //    this->sensor_params.sensor_fy = 581.8;
  //    this->sensor_params.sensor_cx = 308.8;
  //    this->sensor_params.sensor_cy = 253.0;
  //  } else if (calib_mode == 3) {
  //    // TUM Fr3
  //    this->sensor_params.sensor_fx = 567.6;
  //    this->sensor_params.sensor_fy = 570.2;
  //    this->sensor_params.sensor_cx = 324.7;
  //    this->sensor_params.sensor_cy = 250.1;
  //  } else if (calib_mode == 5) {
  //    // ScanNet scene0000_00
  //    this->sensor_params.sensor_fx = 571.623718;
  //    this->sensor_params.sensor_fy = 571.623718;
  //    this->sensor_params.sensor_cx = 319.500000;
  //    this->sensor_params.sensor_cy = 239.500000;
  //    this->sensor_params.sensor_scale = 1000.0f;

  //    this->depth2color_mat.set_identity();
  //    this->depth2color_mat(0, 0) = 0.999973;
  //    this->depth2color_mat(0, 1) = -0.006768;
  //    this->depth2color_mat(0, 2) = -0.002833;
  //    this->depth2color_mat(0, 3) = 0.037800;
  //    this->depth2color_mat(1, 0) = 0.006790;
  //    this->depth2color_mat(1, 1) = 0.999942;
  //    this->depth2color_mat(1, 2) = 0.008347;
  //    this->depth2color_mat(1, 3) = 0.003850;
  //    this->depth2color_mat(2, 0) = 0.002776;
  //    this->depth2color_mat(2, 1) = -0.008366;
  //    this->depth2color_mat(2, 2) = 0.999961;
  //    this->depth2color_mat(2, 3) = 0.022000;
  //        0.999973 -0.006768 -0.002833 0.037800
  //        0.006790 0.999942 0.008347 0.003850
  //        0.002776 -0.008366 0.999961 0.022000
  //        0         0         0        1
  //  }

//  this->sensor_params.sensor_noise_ratio = 0.01f;
//  this->sensor_params.min_range = 0.10f;
//  this->sensor_params.max_range = 8.0f;

  this->depth_params.sensor_noise_ratio = 0.01f;
  this->depth_params.min_range = 0.10f;
  this->depth_params.max_range = 8.0f;

  this->number_of_hierarchy_layers = 3;

  for (int i = 0; i < MAX_LAYER_NUMBER; i++) this->max_iterate_times[i] = 20;

  this->convergence_threshold = 1e-4f;
  this->failure_threshold = 5e-1f;

  this->presegment_cell_width = 16;
  this->pixel_data_weight = 0.1f;
  this->normal_position_weight = 0.8f;
}

void SLAM_system_settings::set_depth_image_size(const int &width,
                                                const int &height,
                                                bool align_flag) {
  this->aligned_depth_size.width =
      ceil_by_stride(width, this->image_alginment_patch_width);
  this->aligned_depth_size.height =
      ceil_by_stride(height, this->image_alginment_patch_width);
}

void SLAM_system_settings::set_color_image_size(const int &width,
                                                const int &height,
                                                bool align_flag) {
  this->aligned_color_size.width =
      ceil_by_stride(width, this->image_alginment_patch_width);
  this->aligned_color_size.height =
      ceil_by_stride(height, this->image_alginment_patch_width);
}

void SLAM_system_settings::set_calibration_paramters(const float &fx,
                                                     const float &fy,
                                                     const float &cx,
                                                     const float &cy,
                                                     const float &scale) {
//  this->sensor_params.sensor_fx = fx;
//  this->sensor_params.sensor_cy = fy;
//  this->sensor_params.sensor_fx = cx;
//  this->sensor_params.sensor_cy = cy;
//  this->sensor_params.sensor_scale = scale;
  this->depth_params.sensor_fx = fx;
  this->depth_params.sensor_cy = fy;
  this->depth_params.sensor_fx = cx;
  this->depth_params.sensor_cy = cy;
  this->depth_params.sensor_scale = scale;
}

void SLAM_system_settings::set_intrinsic(const float &fx, const float &fy,
                                         const float &cx, const float &cy) {
  this->color_params.sensor_fx = fx;
  this->color_params.sensor_fy = fy;
  this->color_params.sensor_cx = cx;
  this->color_params.sensor_cy = cy;
}

void SLAM_system_settings::set_intrinsic(const float &fx, const float &fy,
                                         const float &cx, const float &cy,
                                         const float &scale) {
  this->depth_params.sensor_fx = fx;
  this->depth_params.sensor_fy = fy;
  this->depth_params.sensor_cx = cx;
  this->depth_params.sensor_cy = cy;
  this->depth_params.sensor_scale = scale;
  LOG_WARNING("scale in set_intrinsic");
  LOG_WARNING(this->depth_params.sensor_scale);
}

void SLAM_system_settings::set_extrinsic(const My_Type::Matrix44f d2c,
                                         const My_Type::Matrix44f d2i) {
  this->depth2color_mat = d2c;
  this->depth2imu_mat = d2i;
}
