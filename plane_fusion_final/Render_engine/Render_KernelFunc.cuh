#include <cuda.h>
#include <cuda_runtime.h>

// voxel
#include "Map_engine/voxel_definition.h"

// My type
#include "OurLib/My_matrix.h"

// Render as Phong type
void render_gypsum_CUDA(dim3 block_rect, dim3 thread_rect,
                        My_Type::Vector3f *points_normal,
                        My_Type::Vector4uc *points_color);

// Render voxel weight
void render_weight_CUDA(dim3 block_rect, dim3 thread_rect,
                        My_Type::Vector3f *points_normal, int *points_weight,
                        My_Type::Vector4uc *points_color);

// Render plane label
void render_plane_label_CUDA(dim3 block_rect, dim3 thread_rect,
                             My_Type::Vector3f *points_normal,
                             int *plane_labels,
                             My_Type::Vector4uc *points_color);
//
void render_plane_label_CUDA(dim3 block_rect, dim3 thread_rect,
                             int *plane_labels,
                             My_Type::Vector4uc *points_color);

// ----------
// Reduce depth image range
void reduce_range_CUDA(dim3 block_rect, dim3 thread_rect,
                       const My_Type::Vector3f *dev_raw_aligned_points,
                       int *min_depth, int *max_depth);
// Pseudo render depth image
void pseudo_render_depth_CUDA(dim3 block_rect, dim3 thread_rect,
                              const My_Type::Vector3f *dev_raw_aligned_points,
                              const int *min_depth, const int *max_depth,
                              My_Type::Vector4uc *color_buffer);

// Generate line segments for normal vector
void generate_line_segment_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *dev_raw_aligned_points,
                                const My_Type::Vector3f *dev_normals,
                                My_Type::Line_segment *dev_line_segments);
