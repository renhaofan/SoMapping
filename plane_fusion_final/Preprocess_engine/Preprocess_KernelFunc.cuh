/**
 *  @file Preprocess_KernelFunc.cuh
 *  @brief CUDA kernel function for frame preprocess.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @todo Bilateral filter BUGS.
 */

#include <math.h>

#include "Hierarchy_image.h"
#include "OurLib/My_matrix.h"
#include "SLAM_system/SLAM_system_settings.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

/**
 * @brief Generate aligned float-type raw depth image (aligned image).
 * @param block_rect
 * @param thread_rect
 * @param raw_depth
 * @param sensor_params
 * @param raw_depth_size
 * @param float_type_depth
 */
void generate_float_type_depth_CUDA(dim3 block_rect, dim3 thread_rect,
                                    const RawDepthType *raw_depth,
                                    Sensor_params sensor_params,
                                    My_Type::Vector2i raw_depth_size,
                                    float *float_type_depth);

/**
 * @brief Bilateral filter (3x3)
 * @param block_rect
 * @param thread_rect
 * @param src_depth
 * @param dst_depth
 */
void bilateral_filter_3x3_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float *dst_depth);

/**
 * @brief Bilateral filter (5x5)
 * @param block_rect
 * @param thread_rect
 * @param src_depth
 * @param dst_depth
 */
void bilateral_filter_5x5_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float *dst_depth);

/**
 * @brief Generate aligned points image from filtered depth image.
 * @param block_rect
 * @param thread_rect
 * @param filtered_depth
 * @param sensor_params
 * @param points_image
 */
void generate_aligned_points_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                        const float *filtered_depth,
                                        Sensor_params sensor_params,
                                        My_Type::Vector3f *points_image);

/**
 * @brief Generate aligned intensity image from raw color image.
 * @param block_rect
 * @param thread_rect
 * @param raw_color
 * @param raw_color_size
 * @param aligned_intensity_image
 */
void generate_intensity_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                   const RawColorType *raw_color,
                                   My_Type::Vector2i raw_color_size,
                                   float *aligned_intensity_image);

/**
 * @brief Generate gradient image.
 * @param block_rect
 * @param thread_rect
 * @param aligned_intensity_image
 * @param gradient_image
 */
void generate_gradient_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                  const float *aligned_intensity_image,
                                  My_Type::Vector2f *gradient_image);

/**
 * @brief Compute normals image.
 * @param block_rect
 * @param thread_rect
 * @param points_image
 * @param normals_image
 */
void compute_normals_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *points_image,
                                My_Type::Vector3f *normals_image);

/**
 * @brief Generate hierarchy intensity image(Not sure).
 * @param block_rect
 * @param thread_rect
 * @param src_layer
 * @param src_layer_size
 * @param dst_layer
 */
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector2f *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       My_Type::Vector2f *dst_layer);

/**
 * @brief Generate hierarchy normal and points image.
 * @param block_rect
 * @param thread_rect
 * @param src_layer
 * @param src_layer_size
 * @param dst_layer
 */
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector3f *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       My_Type::Vector3f *dst_layer);

/**
 * @brief down_sample_hierarchy_layers_CUDA
 * @param block_rect
 * @param thread_rect
 * @param src_layer
 * @param src_layer_size
 * @param dst_layer
 */
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const float *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       float *dst_layer);

/**
 * @brief down_sample_hierarchy_layers_CUDA
 * @param block_rect
 * @param thread_rect
 * @param src_layer
 * @param src_layer_size
 * @param dst_layer
 */
template <typename T>
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const T *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       T *dst_layer);

/**
 * @brief outlier_point_filter_CUDA
 * @param block_rect
 * @param thread_rect
 * @param src_depth
 * @param dst_depth
 */
void outlier_point_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float *dst_depth);

/**
 * @brief Bilateral filter (RADIUS*RADIUS)
 * @param block_rect
 * @param thread_rect
 * @param src_depth
 * @param dst_depth
 */
void bilateral_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                           const float *src_depth, float *dst_depth);
