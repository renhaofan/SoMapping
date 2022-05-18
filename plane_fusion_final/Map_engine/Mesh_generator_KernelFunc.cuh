



// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <math.h>

// voxel        
#include "voxel_definition.h"
#include "Plane_detector/Plane_structure.h"
// My type
#include "OurLib/My_matrix.h"



//! Find allocated entries
/*!

*/
void find_alllocated_entries_CUDA(dim3 block_rect, dim3 thread_rect,
								  const HashEntry * entries,
								  int max_number_of_entries,
								  HashEntry * allocated_entries,
								  int * number_of_allocated_entries);


//! Generate triangle mesh from allocated voxel blocks (allocated entries)
/*!

*/
void generate_triangle_mesh_CUDA(dim3 block_rect, dim3 thread_rect,
								 const HashEntry * entries,
								 const HashEntry * allocated_entries,
								 const Voxel_f * voxel_block_array,
								 My_Type::Vector3f * triangles,
								 int * number_of_triangles,
								 int max_number_of_triangles);


//! Generate noraml vectors of each vertex
/*!


*/
void generate_vertex_normals_CUDA(dim3 block_rect, dim3 thread_rect,
								  const My_Type::Vector3f * vertex_array,
								  const HashEntry * entries,
								  const Voxel_f * voxel_block_array,
								  My_Type::Vector3f * vertex_normals);


//! Generate pseudo color array of plane labels
/*!


*/
void generate_vertex_color_CUDA(dim3 block_rect, dim3 thread_rect,
								const My_Type::Vector3f * vertex_array,
								int number_of_vertex,
								const HashEntry * entries,
								const Voxel_f * voxel_block_array,
								My_Type::Vector4uc * vertex_color_array);
void generate_vertex_color_CUDA(dim3 block_rect, dim3 thread_rect,
								const My_Type::Vector3f * vertex_array,
								int number_of_vertex,
								const HashEntry * entries,
								const Voxel_f * voxel_block_array,
								const My_Type::Vector2i * plane_label_mapper,
								My_Type::Vector4uc * vertex_color_array);

//! Find non-planar voxel blocks (hash entry)
/*!

*/

void find_nonplanar_blocks_CUDA(dim3 block_rect, dim3 thread_rect,
								const HashEntry * entries,
								const HashEntry * allocated_entries,
								const Voxel_f * voxel_block_array,
								HashEntry * nonplanar_entries,
								int * number_of_nonplanar_block);


//! Generate triangle vertex from plane
/*!

*/

void generate_triangle_mesh_from_plane_CUDA(dim3 block_rect, dim3 thread_rect,
											Plane_info model_plane,
											Plane_coordinate plane_coordinate,
											const PlaneHashEntry * plane_entries,
											const Plane_pixel * plane_pixel_array,
											My_Type::Vector3f * vertex_array,
											int * triangle_counter);

//!
/*!


*/
void copy_mesh_to_global_map_CUDA(dim3 block_rect, dim3 thread_rect,
								  const My_Type::Vector3f * src_vertex_array,
								  const My_Type::Vector3f * src_normal_array,
								  const My_Type::Matrix44f submap_pose,
								  int number_of_vertex,
								  My_Type::Vector3f * dst_vertex_array,
								  My_Type::Vector3f * dst_normal_array);


//! 
/*!

*/
void generate_vertex_color_CUDA(dim3 block_rect, dim3 thread_rect,
								const My_Type::Vector3f * vertex_array,
								const HashEntry * entries,
								const Voxel_f * voxel_block_array,
								const My_Type::Vector2i * global_plane_id_list,
								My_Type::Vector4uc * vertex_color_array);