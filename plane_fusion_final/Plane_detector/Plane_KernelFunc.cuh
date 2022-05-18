

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// My type
#include "OurLib/My_matrix.h"
#include "Preprocess_engine/Hierarchy_image.h"
//
#include "SLAM_system/SLAM_system_settings.h"

//
#include "Plane_structure.h"

//! Fit plane for each cell
/*!

*/
void fit_plane_for_cells_CUDA(dim3 block_rect, dim3 thread_rect,
                              const My_Type::Vector3f *current_points,
                              const My_Type::Vector3f *current_normals,
                              Sensor_params sensor_params,
                              Cell_info *cell_info_mat);

//! Generate histogram on PxPy
/*!

*/
void histogram_PxPy_CUDA(dim3 block_rect, dim3 thread_rect,
                         Cell_info *cell_info_mat, float *hist_PxPy);

//! Find peaks on PxPy histogram
/*!

*/
void find_PxPy_peaks_CUDA(dim3 block_rect, dim3 thread_rect, float *hist_mat,
                          Hist_normal *hist_normal, int *peak_counter);

//! Generate histogram of projection distance
/*!

*/
void histogram_prj_dist_CUDA(dim3 block_rect, dim3 thread_rect,
                             const Cell_info *cell_info_mat,
                             const Hist_normal *hist_normal,
                             float *prj_dist_hist);

//! Find peaks on distance histogram
/*!

*/
void find_prj_dist_peaks_CUDA(dim3 block_rect, dim3 thread_rect,
                              const float *prj_dist_hist,
                              const Hist_normal *hist_normal, int *peak_index,
                              Plane_info *current_planes);

//! GPU-based K-means iteration
/*!

*/
void K_mean_iterate_CUDA(dim3 block_rect, dim3 thread_rect,
                         Plane_info *current_plane, Cell_info *cell_info_mat,
                         Cell_info *plane_mean_paramters,
                         float *ATA_upper_buffer, float *ATb_buffer,
                         int plane_num);

//! Label current planes
/*!


*/
void label_current_planes_CUDA(dim3 block_rect, dim3 thread_rect,
                               const Cell_info *cell_info_mat,
                               int *current_plane_labels);

//! Count pxiel number of each plane
/*!

*/
void count_planar_pixel_number_CUDA(dim3 block_rect, dim3 thread_rect,
                                    const int *plane_labels,
                                    Plane_info *plane_list, int plane_counter);

//!
/*!

*/
void count_overlap_pxiel_number_CUDA(dim3 block_rect, dim3 thread_rect,
                                     const int *current_plane_labels,
                                     const int *model_plane_labels,
                                     int current_plane_counter,
                                     int *relative_matrix);