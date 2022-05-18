

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <float.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "OurLib/device_color_table.cuh"
#include "OurLib/reduction_KernelFunc.cuh"
#include "Render_KernelFunc.cuh"

//__constant__ float base_color[3] = { 1.0, 0.9, 0.7 };
__constant__ float base_color[3] = {1.0, 1.0, 1.0};

//
__global__ void render_gypsum_KernelFunc(My_Type::Vector3f *points_normal,
                                         My_Type::Vector4uc *points_color) {
  //
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  //
  int index = u + image_width * v;

  //
  My_Type::Vector3f normal;
  normal = points_normal[index];

  //
  bool valid_normal = true;
  if (normal.norm() <= 0.5)
    valid_normal = false;

  //
  My_Type::Vector4uc color;
  // R G B
  if (normal.z > 0)
    normal.z = -normal.z;
  unsigned char color_uc = (unsigned char)max((1.0 - normal.z) * 127.0, 0.0f);
  // unsigned char color_uc = (unsigned char)max((0 - normal.z) * 255.0, 0.0f);
  color.r = (float)color_uc * base_color[0];
  color.g = (float)color_uc * base_color[1];
  color.b = (float)color_uc * base_color[2];
  // Alpha
  if (valid_normal) {
    color.a = 0xFF;
  } else {
    color.a = 0x00;
  }

  //
  points_color[index] = color;
}
// Cpp call CUDA
void render_gypsum_CUDA(dim3 block_rect, dim3 thread_rect,
                        My_Type::Vector3f *points_normal,
                        My_Type::Vector4uc *points_color) {
  render_gypsum_KernelFunc<<<block_rect, thread_rect>>>(points_normal,
                                                        points_color);
}

//
__global__ void render_weight_KernelFunc(My_Type::Vector3f *points_normal,
                                         int *points_weight,
                                         My_Type::Vector4uc *points_color) {
  //
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  //
  int index = u + image_width * v;

  //
  My_Type::Vector3f normal;
  normal = points_normal[index];
  //
  float weight = points_weight[index];

  //
  My_Type::Vector4uc color;
  // R G B
  if (normal.z > 0)
    normal.z = -normal.z;
  color.r = (unsigned char)max(
      128.0 * (1.0 - weight / MAX_SDF_WEIGHT) + (1.0 - normal.z) * 64.0, 0.0f);
  color.g = (unsigned char)max(
      128.0 * weight / MAX_SDF_WEIGHT + (1.0 - normal.z) * 64.0, 0.0f);
  color.b = 0;
  // Alpha
  color.a = 0xFF;

  //
  points_color[index] = color;
}
// Cpp call CUDA
void render_weight_CUDA(dim3 block_rect, dim3 thread_rect,
                        My_Type::Vector3f *points_normal, int *points_weight,
                        My_Type::Vector4uc *points_color) {

  render_weight_KernelFunc<<<block_rect, thread_rect>>>(
      points_normal, points_weight, points_color);
}

//    Label
__global__ void
render_plane_label_KernelFunc(My_Type::Vector3f *points_normal,
                              int *plane_labels,
                              My_Type::Vector4uc *points_color) {
  //
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  //
  int index = u + image_width * v;

  //
  My_Type::Vector3f normal;
  normal = points_normal[index];
  if (normal.z > 0)
    normal.z = -normal.z;
  //     Label
  int plane_label = plane_labels[index];

  //
  My_Type::Vector4uc color;
  //
  if (plane_label > 0) {
    unsigned int color_i = (unsigned int)max((1.0 - normal.z) * 128.0, 0.0f);
    int plane_color_index = (plane_label * PLANE_COLOR_STEP) % COLOR_NUM;

    // Alpha
    color.a = 0xFF;
    // B G R
    color.b =
        (unsigned char)((color_i + color_table_Blue[plane_color_index]) >> 1);
    color.g =
        (unsigned char)((color_i + color_table_Green[plane_color_index]) >> 1);
    color.r =
        (unsigned char)((color_i + color_table_Red[plane_color_index]) >> 1);
  } else {
    // R G B
    unsigned char color_uc = (unsigned char)max((1.0 - normal.z) * 128.0, 0.0f);
    color.r = color_uc;
    color.g = color_uc;
    color.b = color_uc;
    // Alpha
    color.a = 0xFF;
  }

  //
  points_color[index] = color;
}
// Cpp call CUDA
void render_plane_label_CUDA(dim3 block_rect, dim3 thread_rect,
                             My_Type::Vector3f *points_normal,
                             int *plane_labels,
                             My_Type::Vector4uc *points_color) {

  render_plane_label_KernelFunc<<<block_rect, thread_rect>>>(
      points_normal, plane_labels, points_color);
}

//
__global__ void
render_plane_label_KernelFunc(int *plane_labels,
                              My_Type::Vector4uc *points_color) {
  //#define PRIME_X		73856093u
  //#define PRIME_Y		19349669u
  //
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  //
  int index = u + gridDim.x * blockDim.x * v;
  int plane_label = plane_labels[index];

  //
  My_Type::Vector4uc color;
  //
  if (plane_label > 0) {
    int plane_color_index = (plane_label * PLANE_COLOR_STEP) % COLOR_NUM;
    // int plane_color_index = (plane_label ^ 73856093u * plane_label ^
    // 19349669u) % COLOR_NUM;

    // Alpha
    color.a = 0xFF;
    // B G R
    color.b = (unsigned char)color_table_Blue[plane_color_index];
    color.g = (unsigned char)color_table_Green[plane_color_index];
    color.r = (unsigned char)color_table_Red[plane_color_index];
  } else {
    // R G B
    color.r = 0x7F;
    color.g = 0x7F;
    color.b = 0x7F;
    // Alpha
    color.a = 0xFF;
  }

  //
  points_color[index] = color;
}
// Cpp call CUDA
void render_plane_label_CUDA(dim3 block_rect, dim3 thread_rect,
                             int *plane_labels,
                             My_Type::Vector4uc *points_color) {

  render_plane_label_KernelFunc<<<block_rect, thread_rect>>>(plane_labels,
                                                             points_color);
}

__global__ void
reduce_range_KernelFunc(const My_Type::Vector3f *dev_raw_aligned_points,
                        int *min_depth, int *max_depth) {
  bool is_valid_depth = true;

  // Pixel coordinate
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  // Compute index
  int index = u + image_width * v;

  //
  float depth_value = dev_raw_aligned_points[index].z;
  if (depth_value <= FLT_EPSILON)
    is_valid_depth = false;

  // Reduce minimum and maximum
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  __shared__ float min_cache[256], max_cache[256];
  if (is_valid_depth) {
    min_cache[tid] = depth_value;
    max_cache[tid] = depth_value;
  } else {
    min_cache[tid] = FLT_MAX;
    max_cache[tid] = 0;
  }
  //
  block_256_reduce_min(min_cache, tid);
  block_256_reduce_max(max_cache, tid);

  //
  if (tid == 0) {
    atomicMin(min_depth, (int)(1000.0f * min_cache[0]));
    atomicMax(max_depth, (int)(1000.0f * max_cache[0]));
  }
}
// Cpp call CUDA
void reduce_range_CUDA(dim3 block_rect, dim3 thread_rect,
                       const My_Type::Vector3f *dev_raw_aligned_points,
                       int *min_depth, int *max_depth) {

  reduce_range_KernelFunc<<<block_rect, thread_rect>>>(dev_raw_aligned_points,
                                                       min_depth, max_depth);
}

//
__global__ void
pseudo_render_depth_KernelFunc(const My_Type::Vector3f *dev_raw_aligned_points,
                               const int *min_depth, const int *max_depth,
                               My_Type::Vector4uc *color_buffer) {
  bool is_valid_depth = true;

  // Pixel coordinate
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  int index = u + image_width * v;

  // Read depth value
  float depth_value =
      dev_raw_aligned_points[index].z * 1000.0f; /* millimeter */
  if (depth_value == 0.0f)
    is_valid_depth = false;

  //
  My_Type::Vector4uc temp_color((unsigned char)0x00);
  if (is_valid_depth) {
    int color_index =
        (int)((depth_value - (float)min_depth[0]) /
              (float)(max_depth[0] - min_depth[0]) * (float)(COLOR_NUM));

    //
    temp_color.r = (unsigned char)color_table_Red[color_index];
    temp_color.g = (unsigned char)color_table_Green[color_index];
    temp_color.b = (unsigned char)color_table_Blue[color_index];
    temp_color.a = 0xFF;
  }

  color_buffer[index] = temp_color;
}
//
void pseudo_render_depth_CUDA(dim3 block_rect, dim3 thread_rect,
                              const My_Type::Vector3f *dev_raw_aligned_points,
                              const int *min_depth, const int *max_depth,
                              My_Type::Vector4uc *color_buffer) {
  pseudo_render_depth_KernelFunc<<<block_rect, thread_rect>>>(
      dev_raw_aligned_points, min_depth, max_depth, color_buffer);
}

// The segment line length of normal vector
#define NORMAL_LINE_SEGMENT_LENGTH 0.02f
//
__global__ void generate_line_segment_KernelFunc(
    const My_Type::Vector3f *dev_raw_aligned_points,
    const My_Type::Vector3f *dev_normals,
    My_Type::Line_segment *dev_line_segments) {
  bool is_valid_depth = true;

  // Pixel coordinate
  int u, v, image_width;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  image_width = gridDim.x * blockDim.x;
  int index = u + image_width * v;

  // Read point
  My_Type::Vector3f temp_point = dev_raw_aligned_points[index];
  if (temp_point.z == 0.0f)
    is_valid_depth = false;

  //
  My_Type::Line_segment temp_line_segment;
  temp_line_segment.origin = 0.0f;
  temp_line_segment.dst = 0.0f;
  if (is_valid_depth) {
    temp_line_segment.origin = temp_point;
    temp_line_segment.dst =
        temp_point + NORMAL_LINE_SEGMENT_LENGTH * dev_normals[index];
  }

  //
  dev_line_segments[index] = temp_line_segment;
}
//
void generate_line_segment_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *dev_raw_aligned_points,
                                const My_Type::Vector3f *dev_normals,
                                My_Type::Line_segment *dev_line_segments) {

  generate_line_segment_KernelFunc<<<block_rect, thread_rect>>>(
      dev_raw_aligned_points, dev_normals, dev_line_segments);
}
