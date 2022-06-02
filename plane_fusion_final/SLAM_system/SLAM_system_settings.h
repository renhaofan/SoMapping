/**
 *  @file SLAM_system_settings.h
 *  @brief Sensor info, tracker info, plane parameters and whether enable plane,
 *         whether generate mesh for vis.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @note Macro for different os
 *        https://iq.opengenus.org/detect-operating-system-in-c/
 *  @todo Check CUDA error. Many times comment CUDA_CKECK_KERNEL. (NOTE)
 */

#pragma once

#include "OurLib/My_matrix.h"
// extract MAX_LAYER_NUMBER and COMPILE_DEBUG_CODE
#include "Config.h"

//#define CUDA_CKECK_KERNEL
/** @brief Check CUDA error. Many times comment CUDA_CKECK_KERNEL. (NOTE)*/
#define CUDA_CKECK_KERNEL checkCudaErrors(cuCtxSynchronize());

/**
 * @brief Depth Sensor parameters, such as fx,fy,cx,cy,depth range, scale, etc.
 */
typedef struct _Sensor_params {
  /** @brief fx parameter. */
  float sensor_fx;
  /** @brief fy parameter. */
  float sensor_fy;
  /** @brief cx parameter. */
  float sensor_cx;
  /** @brief cy parameter. */
  float sensor_cy;
  /** @brief Sensor scale (For example: Kinect scale = 1.0m / 1.0mm = 1000.0f).
   */
  float sensor_scale;
  /** @brief Sensor noise radius per meter. */
  float sensor_noise_ratio;
  /** @brief Sensor min range. */
  float min_range;
  /** @brief Sensor max range. */
  float max_range;
} Sensor_params;

class SLAM_system_settings {
 public:
  /** @brief The pointer to this static object. */
  static SLAM_system_settings *instance_ptr;
  /** @brief Generate a instance for this static object. */
  static SLAM_system_settings *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new SLAM_system_settings();
    return instance_ptr;
  }

  /** @brief Flag whether generate mesh for visualization. */
  bool generate_mesh_for_visualization;
  /** @brief Flag whether use plane prior info. */
  bool enable_plane_module;
#if __unix__
#pragma region "Image / Sensor parameters" {
#elif _WIN32
#pragma region(Image / Sensor parameters)
#endif
  /**
   *  @brief CUDA memory alignment size.
   *  @warning Must be set in range of {1, 2, 4, ..., 16}.
   */
  int image_alginment_patch_width;
  /**
   *  @brief Raycast range patch size.
   *  @warning No bigger than this->image_alginment_patch_width!
   */
  int raycast_range_patch_width;

  /** @brief Aligned depth image size. */
  My_Type::Vector2i aligned_depth_size;
  /** @brief Aligned color image size. */
  My_Type::Vector2i aligned_color_size;

  /** @brief Sensor parameters. */
  Sensor_params sensor_params;
  /** @brief Color sensor parameters. */
  Sensor_params color_params;
  /** @brief Depth sensor parameters. */
  Sensor_params depth_params;
  /** @brief Extrinsics from depth sensor to color sensor. */
  My_Type::Matrix44f depth2color_mat;
  /** @brief Extrinsics from depth sensor to imu sensor. */
  My_Type::Matrix44f depth2imu_mat;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Plane detector parameters" {
#elif _WIN32
#pragma region(Plane detector parameters)
#endif
  /**
   * @brief Pre-segment cell width.
   * @warning Must be the common divisor of aligned_depth_size.
   */
  int presegment_cell_width;
  /** @brief Weight coefficients of super pixel cluster. */
  float pixel_data_weight;
  /** @brief Weight coefficients of super pixel cluster. */
  float normal_position_weight;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Tracker parameters" {
#elif _WIN32
#pragma region(Tracker parameters)
#endif
  /** @brief The number of hierarchy layers. */
  int number_of_hierarchy_layers;
  /** @brief Max iterate times of each layer. */
  int max_iterate_times[MAX_LAYER_NUMBER];
  /** @brief Convergence threshold of nonlinear solver (Tracker). */
  float convergence_threshold;
  /** @brief Failed threshold of nonlinear solver (Tracker). */
  float failure_threshold;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  /** @brief Default constructor. Call this->set_to_default() */
  SLAM_system_settings();
  /** @brief Default dstructor. */
  ~SLAM_system_settings();

  /** @brief Set default values for parameters. */
  void set_to_default();
  /**
   * @brief Set color camera intrinsic matrix.
   * @param fx Camera fx.
   * @param fy Camera fy.
   * @param cx Camera cx.
   * @param cy Camera cy.
   */
  void set_intrinsic(const float &fx, const float &fy, const float &cx,
                     const float &cy);
  /**
   * @brief Set depth image size.
   * @param width Assigned depth image width.
   * @param height Assigned depth image height.
   * @param align_flag Flag whether alignment, true by default.
   */
  void set_depth_image_size(const int &width, const int &height,
                            bool align_flag = true);
  /**
   * @brief Set color image size.
   * @param width Assigned color image width.
   * @param height Assigned color image height.
   * @param align_flag Flag whether alignment, true by default.
   */
  void set_color_image_size(const int &width, const int &height,
                            bool align_flag = true);
  /**
   * @brief Set Calibration Paramters.
   * @param fx Camera fx.
   * @param fy Camera fy.
   * @param cx Camera cx.
   * @param cy Camera cy.
   * @param scale Camera scale.
   */
  void set_calibration_paramters(const float &fx, const float &fy,
                                 const float &cx, const float &cy,
                                 const float &scale);
  /**
   * @brief Set depth camera intrinsic matrix.
   * @param fx Depth camera fx.
   * @param fy Depth camera fy.
   * @param cx Depth camera cx.
   * @param cy Depth camera cy.
   * @param scale Depth Camera scale.
   */
  void set_intrinsic(const float &fx, const float &fy, const float &cx,
                     const float &cy, const float &scale);
  /**
   * @brief Set extrinsic matrix.
   * @param d2c Extrinsic from depth sensor to color sensor.
   * @param d2i Extrinsic from depth sensor to imu sensor.
   */
  void set_extrinsic(const My_Type::Matrix44f d2c,
                     const My_Type::Matrix44f d2i);
};
