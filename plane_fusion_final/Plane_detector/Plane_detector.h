/**
 *  @file Plane_detector.h
 *  @brief TODO
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

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
  /** @brief Size of aligned depth image. */
  My_Type::Vector2i aligned_depth_size;
  /** @brief Size of cells matrix. */
  My_Type::Vector2i cell_mat_size;

  /** @brief Current cell information matrix. */
  Cell_info *cell_info_mat;
  /** @brief Current cell information matrix. */
  Cell_info *dev_cell_info_mat;
  /** @brief Cell label of each pixel/point. */
  int *dev_current_cell_labels;

  /** @brief Current plane_labels. */
  int *dev_current_plane_labels;
  /** @brief Model plane labels (store in Plane_Map). */
  int current_plane_counter;
  /** @brief Current plane counter. */
  int *dev_current_plane_counter;

  /** @brief Plane local coordinate. */
  Plane_coordinate *dev_buffer_coordinate;

  /** @brief Current plane list. */
  Plane_info *current_planes;
  /** @brief Current plane list. */
  Plane_info *dev_current_planes;

  /** @brief Model plane list (store in Plane_Map). */
  Plane_info *model_planes;
  /** @brief Model plane list (store in Plane_Map). */
  Plane_info *dev_model_planes;

  /** @brief For plane matching. */
  int *relative_matrix;
  /** @brief For plane matching. */
  int *dev_relative_matrix;

  /** @brief <current-model> plane matches list. */
  std::vector<My_Type::Vector2i> matches;
  /** @brief <current-model> plane matches list. */
  My_Type::Vector2i *dev_matches;

  StopWatchInterface *timer_average;

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
  /**
   * @brief Prepare to detect planes
   */
  virtual void prepare_to_detect();

  /**
   * @brief Pre-segmentation current points to cells(RegularGrids/SuperPixels)
   * @param dev_current_points
   * @param dev_current_normals
   */
  virtual void presegment_to_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals) = 0;

  /**
   * @brief Fit plane for each cell
   * @param dev_current_points
   * @param dev_current_normals
   * @param dev_cell_info_mat
   */
  virtual void fit_plane_for_each_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals,
      Cell_info *dev_cell_info_mat) = 0;

  /**
   * @brief Cluster cells to planes
   * @param dev_cell_info_mat
   * @param dev_model_planes
   * @param dev_current_planes
   * @param with_continuous_frame_tracking
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
  /** @brief CUDA device pointer of hist_PxPy. */
  float *hist_mat;
  /** @brief CUDA device pointer of hist_PxPy. */
  float *dev_hist_mat;

  /** @brief CUDA device pointer of hist_normal. */
  Hist_normal *dev_hist_normals;
  /** @brief The number of normal vectors obtained by histogram statistics. */
  int hist_normal_counter;
  /** @brief The number of normal vectors obtained by histogram statistics. */
  int *dev_hist_normal_counter;

  /** @brief Plane distance histogram. */
  float *dev_prj_distance_hist;

  /** @brief Some buffers for GPU-based K-means iteration. */
  Cell_info *dev_plane_mean_parameters;

  float *dev_ATA_upper_buffer;
  float *dev_ATb_buffer;

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
                               Cell_info *dev_cell_info_mat) override;

  void cluster_cells(Cell_info *dev_cell_info_mat,
                     const Plane_info *dev_model_planes,
                     Plane_info *dev_current_planes,
                     bool with_continuous_frame_tracking = false) override;
};

class Plane_super_pixel_detector : public Plane_detector {
 public:
  My_Type::Vector2i super_pixel_mat_size;

  int number_of_CUDA_block_per_line;
  int number_of_CUDA_block_per_cell;

  Super_pixel *dev_super_pixel_mat;
  Super_pixel *dev_super_pixel_accumulate_mat;

  int *dev_super_pixel_id_image;

  Plane_coordinate *dev_base_vectors;

  My_Type::Vector3f *dev_cell_hessain_uppers;

  My_Type::Vector2f *dev_cell_nabla;

  bool *super_pixel_adjacent_mat;

  Plane_super_pixel_detector();
  ~Plane_super_pixel_detector();

  void init() override;

 private:
  void prepare_to_detect() override;

  void presegment_to_cell(
      const My_Type::Vector3f *dev_current_points,
      const My_Type::Vector3f *dev_current_normals) override;

  void fit_plane_for_each_cell(const My_Type::Vector3f *dev_current_points,
                               const My_Type::Vector3f *dev_current_normals,
                               Cell_info *dev_cell_info_mat) override;

  void cluster_cells(Cell_info *dev_cell_info_mat,
                     const Plane_info *dev_model_planes,
                     Plane_info *dev_current_planes,
                     bool with_continuous_frame_tracking = false) override;
};
