




// My type
#include "Hierarchy_image.h"
#include "OurLib/My_matrix.h"
#include <math.h>
//
#include "SLAM_system/SLAM_system_settings.h"


// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>


//! Generate aligned float-type raw depth image (aligned image)
/*!
	\params	block_rect	

	\params	thread_rect

	\params	raw_depth

	\params	sensor_params

	\params	raw_depth_size

	\params	float_type_depth

	\return void
*/
void generate_float_type_depth_CUDA(dim3 block_rect, dim3 thread_rect,
									const RawDepthType * raw_depth,
									Sensor_params sensor_params,
									My_Type::Vector2i raw_depth_size,
									float * float_type_depth);


//! Bilateral filter (3x3)
/*!

*/
void bilateral_filter_3x3_CUDA(dim3 block_rect, dim3 thread_rect,
							   const float * src_depth,
							   float * dst_depth);


//! Bilateral filter (5x5)
/*!

*/
void bilateral_filter_5x5_CUDA(dim3 block_rect, dim3 thread_rect,
							   const float * src_depth,
							   float * dst_depth);


//! Generate aligned points image from filtered depth image. 
/*!

*/
void generate_aligned_points_image_CUDA(dim3 block_rect, dim3 thread_rect,
										const float * filtered_depth,
										Sensor_params sensor_params,
										My_Type::Vector3f * points_image);


//! Generate aligned intensity image from raw color image
/*!


*/
void generate_intensity_image_CUDA(dim3 block_rect, dim3 thread_rect,
								   const RawColorType * raw_color,
								   My_Type::Vector2i raw_color_size,
								   float * aligned_intensity_image);


//! Generate gradient image
/*!

*/
void generate_gradient_image_CUDA(dim3 block_rect, dim3 thread_rect,
								  const float * aligned_intensity_image,
								  My_Type::Vector2f * gradient_image);


//! Compute normals image
/*!

*/
void compute_normals_image_CUDA(dim3 block_rect, dim3 thread_rect,
								const My_Type::Vector3f * points_image,
								My_Type::Vector3f * normals_image);


//! Generate hierarchy normal and points image
/*!

*/
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector2f * src_layer, My_Type::Vector2i src_layer_size,
                                       My_Type::Vector2f * dst_layer);
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector3f * src_layer, My_Type::Vector2i src_layer_size,
                                       My_Type::Vector3f * dst_layer);
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const float * src_layer, My_Type::Vector2i src_layer_size,
                                       float * dst_layer);



template <typename T>
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const T * src_layer, My_Type::Vector2i src_layer_size,
                                       T * dst_layer);






void outlier_point_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float * src_depth,
                               float * dst_depth);

//! Bilateral filter (RADIUS*RADIUS)
/*!

*/
void bilateral_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                           const float * src_depth,
                           float * dst_depth);
