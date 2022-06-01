/**
 *  Copyright (C) All rights reserved.
 *  @file Plane_detector.h
 *  @brief TODO
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

//
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/FFT>
#include <vector>

#include "OurLib/My_matrix.h"
#include "Plane_detector/Plane_structure.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

class Plane_detector {
 public:
  //! Size of aligned depth image
  My_Type::Vector2i aligned_depth_size;
  //! Size of cells matrix
  My_Type::Vector2i cell_mat_size;

  //! Current cell information matrix
  Cell_info *cell_info_mat, *dev_cell_info_mat;
  //! Cell label of each pixel/point
  int *dev_current_cell_labels;

  //! Current plane_labels
  int *dev_current_plane_labels;
  //! Model plane labels (store in Plane_Map)
  // ...
  //! Current plane counter
  int current_plane_counter, *dev_current_plane_counter;

  //! Plane local coordinate
  Plane_coordinate *dev_buffer_coordinate;

  //! Current plane list
  Plane_info *current_planes, *dev_current_planes;
  //! Model plane list (store in Plane_Map)
  Plane_info *model_planes, *dev_model_planes;

  //! (for plane matching)
  int *relative_matrix, *dev_relative_matrix;
  //! <current-model> plane matches list
  std::vector<My_Type::Vector2i> matches;
  My_Type::Vector2i *dev_matches;

  //!
  StopWatchInterface *timer_average;

  //
  std::vector<Plane_info> plane_region_params;
  std::vector<std::vector<int>> planar_cell_index_list;

  Plane_detector();
  ~Plane_detector();

  virtual void init();

  void detect_plane(const My_Type::Vector3f *dev_current_points,
                    const My_Type::Vector3f *dev_current_normals,
                    const Eigen::Matrix4f &camera_pose,
                    const Plane_info *dev_model_planes = nullptr,
                    int *dev_previous_plane_labels = nullptr,
                    bool with_continuous_frame_tracking = false);

  void match_planes(const Plane_info *model_planes, int model_plane_number,
                    const int *dev_current_plane_labels,
                    const int *dev_model_plane_labels);

  void match_planes_to_new_map(const Plane_info *model_planes,
                               int model_plane_number,
                               const int *dev_current_plane_labels,
                               const int *dev_model_plane_labels,
                               std::vector<std::pair<int, int>> &plane_matches);

 protected:
  //! Prepare to detect planes
  /*!

  */
  virtual void prepare_to_detect();

  //! Pre-segmentation current points to cells(RegularGrids/SuperPixels)
  /*!

  */
  virtual void presegment_to_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals) = 0;

  //! Fit plane for each cell
  /*!

  */
  virtual void fit_plane_for_each_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals,
      Cell_info *dev_cell_info_mat) = 0;

  //! Cluster cells to planes
  /*!

  */
  virtual void cluster_cells(Cell_info *dev_cell_info_mat,
                             const Plane_info *dev_model_planes,
                             Plane_info *dev_current_planes,
                             bool with_continuous_frame_tracking = false) = 0;

  void transfer_plane_coordinate(const Eigen::Matrix4f &camera_pose);

  std::vector<std::pair<int, int>> find_most_overlap_model_plane(
      int current_plane_label, int model_plane_counter);
};

class Plane_stereoprojection_detector : public Plane_detector {
 public:
  //! CUDA device pointer of hist_PxPy.
  float *hist_mat, *dev_hist_mat;
  //! CUDA device pointer of hist_normal.
  Hist_normal *dev_hist_normals;
  //! The number of normal vectors obtained by histogram statistics.
  int hist_normal_counter, *dev_hist_normal_counter;

  //! Plane distance histogram
  float *dev_prj_distance_hist;

  //! Some buffers for GPU-based K-means iteration
  Cell_info *dev_plane_mean_parameters;
  float *dev_ATA_upper_buffer, *dev_ATb_buffer;

  //!
  Plane_stereoprojection_detector();
  ~Plane_stereoprojection_detector();

  void init() override;

 protected:
  void prepare_to_detect() override;

  void presegment_to_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals) override{};

  void fit_plane_for_each_cell(const My_Type::Vector3f *dev_current_points,
                               const My_Type::Vector3f *dev_current_normals,
                               Cell_info *dev_cell_info_mat);

  void cluster_cells(Cell_info *dev_cell_info_mat,
                     const Plane_info *dev_model_planes,
                     Plane_info *dev_current_planes,
                     bool with_continuous_frame_tracking = false) override;
};

#define TEST_OLD_METOD 0
//!
/*!

*/
class Plane_super_pixel_detector : public Plane_detector {
 public:
#if (TEST_OLD_METOD)
  //! CUDA device pointer of hist_PxPy.
  float *hist_mat, *dev_hist_mat;
  //! CUDA device pointer of hist_normal.
  Hist_normal *dev_hist_normals;
  //! The number of normal vectors obtained by histogram statistics.
  int hist_normal_counter, *dev_hist_normal_counter;

  //! Plane distance histogram
  float *dev_prj_distance_hist;

  //! Some buffers for GPU-based K-means iteration
  Cell_info *dev_plane_mean_parameters;
  float *dev_ATA_upper_buffer, *dev_ATb_buffer;
#endif

  //!
  My_Type::Vector2i super_pixel_mat_size;
  //
  int number_of_CUDA_block_per_line;
  int number_of_CUDA_block_per_cell;

  //!
  Super_pixel *dev_super_pixel_mat;
  Super_pixel *dev_super_pixel_accumulate_mat;
  //!
  int *dev_super_pixel_id_image;

  //!
  Plane_coordinate *dev_base_vectors;
  //!
  My_Type::Vector3f *dev_cell_hessain_uppers;
  //!
  My_Type::Vector2f *dev_cell_nabla;

  //!
  bool *super_pixel_adjacent_mat;

  //!
  Plane_super_pixel_detector();
  ~Plane_super_pixel_detector();

  //
  void init() override;

 private:
  //
  void prepare_to_detect() override;

  //
  void presegment_to_cell(const My_Type::Vector3f *dev_current_points,
                          const My_Type::Vector3f *dev_current_normals);

  //
  void fit_plane_for_each_cell(const My_Type::Vector3f *dev_current_points,
                               const My_Type::Vector3f *dev_current_normals,
                               Cell_info *dev_cell_info_mat);

  //
  void cluster_cells(Cell_info *dev_cell_info_mat,
                     const Plane_info *dev_model_planes,
                     Plane_info *dev_current_planes,
                     bool with_continuous_frame_tracking = false) override;
};
