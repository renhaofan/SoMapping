

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
#include <float.h>

#include "OurLib/reduction_KernelFunc.cuh"
#include "Plane_detector_KernelFunc.cuh"

// Tool functions
//
__device__ void generate_local_coordinate(Plane_coordinate &local_coordinate);
//
__device__ inline void transform_to_local_coordinate(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate);
//
__device__ inline void transform_from_local_coordinate(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate);

// Fit plane for each patch
__global__ void fit_plane_for_cells_KernelFunc(
    const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *current_normals, Sensor_params sensor_params,
    Cell_info *cell_info_mat) {
  // Coordinate/index of cell
  int u_cell = blockIdx.x;
  int v_cell = blockIdx.y;
  int cell_index = u_cell + v_cell * gridDim.x;
  // Coordinate/index of point
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;
  int index = u + v * gridDim.x * blockDim.x;

  //
  bool is_valid_cell = true;
  // Load and validate point position
  My_Type::Vector3f current_point = current_points[index];
  if (current_point.z < FLT_EPSILON) is_valid_cell = false;
  // Load and validate point normal vector existence
  My_Type::Vector3f current_normal = current_normals[index];
  if (current_normal.norm() < FLT_EPSILON) is_valid_cell = false;

#pragma region(Compute position of points in this cell)

  // Thread id
  int tid = threadIdx.x + blockDim.x * threadIdx.y;

  // Reduction cache buffer (shared memory)
  /*	ToDo :	How to solve "warning : __shared__ memory variable with
  non-empty constructor or destructor (potential race between threads)" see :
  https://stackoverflow.com/questions/27230621/cuda-shared-memory-inconsistent-results
  */
  __shared__ My_Type::Vector3f cache_float3[256];
  if (is_valid_cell) {
    cache_float3[tid] = current_point;
  } else {
    cache_float3[tid] = 0.0f;
  }
  block_256_reduce(cache_float3, tid);

  // Weight center of points in this cell
  My_Type::Vector3f weight_center = cache_float3[0];
  __syncthreads();

  // Counter the number of valid points
  __shared__ float cache_float1[256];
  if (is_valid_cell) {
    cache_float1[tid] = 1.0f;
  } else {
    cache_float1[tid] = 0.0f;
  }
  block_256_reduce(cache_float1, tid);
  //
  float valid_num = cache_float1[0];
  weight_center /= valid_num;

  if (tid == 0 && is_valid_cell) {
    // Save the weight center position
    cell_info_mat[cell_index].x = weight_center.x;
    cell_info_mat[cell_index].y = weight_center.y;
    cell_info_mat[cell_index].z = weight_center.z;
  }

#pragma endregion

#pragma region(Fit normal vector for each cell)

  //
  current_point -= weight_center;

  // ATA = {a0, a1}	|	ATb = {b0}
  //		 {a1, a2}	|		  {b1}
  float ATA_upper[3], ATb[2];

  if (is_valid_cell) {
    //
    ATA_upper[0] = current_point.x * current_point.x;
    ATA_upper[1] = current_point.x * current_point.y;
    ATA_upper[2] = current_point.y * current_point.y;
    //
    ATb[0] = -current_point.x * current_point.z;
    ATb[1] = -current_point.y * current_point.z;
  } else {
    //
    ATA_upper[0] = 0.0f;
    ATA_upper[1] = 0.0f;
    ATA_upper[2] = 0.0f;

    //
    ATb[0] = 0.0f;
    ATb[1] = 0.0f;
  }

  // Reduce ATA and ATb
  // ATA
  //#pragma unroll 1
  for (int i = 0; i < 3; i++) {
    __syncthreads();
    cache_float1[tid] = ATA_upper[i];
    // Reduce
    block_256_reduce(cache_float1, tid);
    if (tid == 0) {
      ATA_upper[i] = cache_float1[0];
    }
  }
  // ATb
  //#pragma unroll 1
  for (int i = 0; i < 2; i++) {
    __syncthreads();
    cache_float1[tid] = ATb[i];
    // Reduce
    block_256_reduce(cache_float1, tid);
    if (tid == 0) {
      ATb[i] = cache_float1[0];
    }
  }

  // normal vector of this cell
  __shared__ My_Type::Vector3f cell_normal_share;
  My_Type::Vector3f cell_normal;
  //    ATA x = ATb
  if (tid == 0 && is_valid_cell) {
    float D, D1, D2, tan_xz, tan_yz;
    // Compute Crammer
    D = ATA_upper[0] * ATA_upper[2] - ATA_upper[1] * ATA_upper[1];
    D1 = ATb[0] * ATA_upper[2] - ATb[1] * ATA_upper[1];
    D2 = ATb[1] * ATA_upper[0] - ATb[0] * ATA_upper[1];
    // compute tangent
    tan_xz = D1 / D;
    tan_yz = D2 / D;

    //
    cell_normal.z = 1 / norm3df(tan_xz, tan_yz, 1.0f);
    cell_normal.x = tan_xz * cell_normal.z;
    cell_normal.y = tan_yz * cell_normal.z;

    //   Ray
    My_Type::Vector3f ray;
    ray.x = ((float)u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
    ray.y = ((float)v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
    ray.z = 1.0f;

    if ((ray.x * cell_normal.x + ray.y * cell_normal.y +
         ray.z * cell_normal.z) > 0.0f) {
      cell_normal.x = -cell_normal.x;
      cell_normal.y = -cell_normal.y;
      cell_normal.z = -cell_normal.z;
    }

    //   Patch
    cell_info_mat[cell_index].nx = cell_normal.x;
    cell_info_mat[cell_index].ny = cell_normal.y;
    cell_info_mat[cell_index].nz = cell_normal.z;

    //        Shared Memory
    cell_normal_share = cell_normal;
  }

#pragma endregion

#pragma region(Counter the number of valid points)

  // Load cell normal vector for each cell
  __syncthreads();
  cell_normal = cell_normal_share;
  float prj_value = cell_normal.x * current_point.x +
                    cell_normal.y * current_point.y +
                    cell_normal.z * current_point.z;
  if (fabsf(prj_value) > 1e-2f) is_valid_cell = false;

  // Reduce Patch
  cache_float1[tid] = 0.0f;
  if (is_valid_cell) {
    cache_float1[tid] = 1.0f;
  }
  // Reduce
  block_256_reduce(cache_float1, tid);
  if (tid == 0) {
    if (is_valid_cell) {
      cell_info_mat[cell_index].counter = (int)cache_float1[0];
    } else {
      cell_info_mat[cell_index].counter = 0;
    }
  }

#pragma endregion
}
// Cpp call CUDA
void fit_plane_for_cells_CUDA(dim3 block_rect, dim3 thread_rect,
                              const My_Type::Vector3f *current_points,
                              const My_Type::Vector3f *current_normals,
                              Sensor_params sensor_params,
                              Cell_info *cell_info_mat) {
  fit_plane_for_cells_KernelFunc<<<block_rect, thread_rect>>>(
      current_points, current_normals, sensor_params, cell_info_mat);
}

//
#define MIN_VALID_POINTS_IN_CELL 50
//#define MIN_VALID_POINTS_IN_CELL	200
//
__global__ void histogram_PxPy_KernelFunc(const Cell_info *cell_info_mat,
                                          float *hist_PxPy) {
  // Max histogram range
  const float max_range = (HISTOGRAM_WIDTH / 2 - 1) * HISTOGRAM_STEP;

  //
  bool is_valid_cell = true;
  // Cell index
  int cell_index = threadIdx.x + blockIdx.x * blockDim.x;

  // Validate cell
  if (cell_info_mat[cell_index].counter < MIN_VALID_POINTS_IN_CELL)
    is_valid_cell = false;

  // Compute histogram coordinate
  float px, py;
  //
  if (is_valid_cell) {
    //
    px = 2.0f * cell_info_mat[cell_index].nx /
         (1.0f - cell_info_mat[cell_index].nz);
    py = 2.0f * cell_info_mat[cell_index].ny /
         (1.0f - cell_info_mat[cell_index].nz);
    //
    if ((px >= max_range) || (px <= -max_range)) is_valid_cell = false;
    if ((py >= max_range) || (py <= -max_range)) is_valid_cell = false;
  }

  //
  if (is_valid_cell) {
    //
    float weighted_area = 1.0f;
    weighted_area = cell_info_mat[cell_index].area;
    //
    float d1 =
        sqrtf(cell_info_mat[cell_index].nx * cell_info_mat[cell_index].nx +
              cell_info_mat[cell_index].ny * cell_info_mat[cell_index].ny);
    float h1 = (1.0f - cell_info_mat[cell_index].nz);
    float theta = acosf(h1 / sqrtf(h1 * h1 + d1 * d1));
    weighted_area *= 4 * (cos(theta) + tan(theta / 2) * sin(theta)) /
                     ((1 - cell_info_mat[cell_index].nz) *
                      (1 - cell_info_mat[cell_index].nz));

    int u_hist, v_hist, index_hist;
    u_hist = (int)roundf((px + max_range) / HISTOGRAM_STEP);
    v_hist = (int)roundf((py + max_range) / HISTOGRAM_STEP);
    index_hist = u_hist + v_hist * HISTOGRAM_WIDTH;

    // 8-neighbor Gaussian sum
    atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH - 1],
              weighted_area / 16.0);
    atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH + 0],
              weighted_area / 8.0);
    atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH + 1],
              weighted_area / 16.0);
    atomicAdd((float *)&hist_PxPy[index_hist - 1], weighted_area / 8.0);
    atomicAdd((float *)&hist_PxPy[index_hist + 0], weighted_area / 4.0);
    atomicAdd((float *)&hist_PxPy[index_hist + 1], weighted_area / 8.0);
    atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH - 1],
              weighted_area / 16.0);
    atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH + 0],
              weighted_area / 8.0);
    atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH + 1],
              weighted_area / 16.0);
  }
}
// Cpp call CUDA
void histogram_PxPy_CUDA(dim3 block_rect, dim3 thread_rect,
                         Cell_info *cell_info_mat, float *hist_PxPy) {
  histogram_PxPy_KernelFunc<<<block_rect, thread_rect>>>(cell_info_mat,
                                                         hist_PxPy);
}

//
__global__ void find_PxPy_peaks_KernelFunc(const float *hist_mat,
                                           Hist_normal *hist_normal,
                                           int *peak_counter) {
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;
  int index = u + v * blockDim.x * gridDim.x;

  //
  float px = (float)(u - (HISTOGRAM_WIDTH >> 1)) * HISTOGRAM_STEP;
  float py = (float)(v - (HISTOGRAM_WIDTH >> 1)) * HISTOGRAM_STEP;
  float prj_radius = sqrtf(px * px + py * py);
  //
  const float max_range = (HISTOGRAM_WIDTH / 2 - 1) * HISTOGRAM_STEP;
  //
  if ((px >= max_range) || (px <= -max_range)) return;
  if ((py >= max_range) || (py <= -max_range)) return;

  //
  float peak_value = hist_mat[index];
  // if (peak_value < MIN_CELLS_OF_DIRECTION / 4.0)			return;
  if (peak_value < MIN_AREA_OF_DIRECTION / 4.0) return;

  // Read neigthbor value
  float neighbor_value[3][3];
  bool is_peak = true;
  //
  neighbor_value[0][0] = hist_mat[index - 1 - blockDim.x * gridDim.x];
  neighbor_value[0][1] = hist_mat[index + 0 - blockDim.x * gridDim.x];
  neighbor_value[0][2] = hist_mat[index + 1 - blockDim.x * gridDim.x];
  neighbor_value[1][0] = hist_mat[index - 1];
  neighbor_value[1][1] = hist_mat[index + 0];
  neighbor_value[1][2] = hist_mat[index + 1];
  neighbor_value[2][0] = hist_mat[index - 1 + blockDim.x * gridDim.x];
  neighbor_value[2][1] = hist_mat[index + 0 + blockDim.x * gridDim.x];
  neighbor_value[2][2] = hist_mat[index + 1 + blockDim.x * gridDim.x];
  // find maximum value
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      if (neighbor_value[i][j] > peak_value) {
        is_peak = false;
        return;
      }
    }

  //
  if (is_peak) {
    int peak_index = atomicAdd(peak_counter, 1);

    //
    if (peak_index < MAX_HIST_NORMALS) {
      //
      float nx, ny, nz, theta, radius_normal;
      theta = atanf(prj_radius / 2.0f);

      // Compute normal vector
      nz = -cos(2 * theta);
      radius_normal = sin(2 * theta);
      nx = px * radius_normal / prj_radius;
      ny = py * radius_normal / prj_radius;

      //
      hist_normal[peak_index].nx = nx;
      hist_normal[peak_index].ny = ny;
      hist_normal[peak_index].nz = nz;
      hist_normal[peak_index].weight = peak_value;

      // printf("N[%d] = (%f, %f, %f) S = %f\r\n", peak_index, nx, ny, nz,
      // peak_value);
    }
  }
}
// Cpp call CUDA
void find_PxPy_peaks_CUDA(dim3 block_rect, dim3 thread_rect, float *hist_mat,
                          Hist_normal *hist_normal, int *peak_counter) {
  find_PxPy_peaks_KernelFunc<<<block_rect, thread_rect>>>(hist_mat, hist_normal,
                                                          peak_counter);
}

//
__global__ void histogram_prj_dist_KernelFunc(const Cell_info *cell_info_mat,
                                              const Hist_normal *hist_normal,
                                              float *distance_histogram) {
  bool is_valid_cell = true;
  int cell_index = threadIdx.x + blockIdx.x * blockDim.x;

  // Validate cell
  int valid_counter = cell_info_mat[cell_index].counter;
  if (valid_counter < MIN_VALID_POINTS_IN_CELL) is_valid_cell = false;

  // Compute included angle between plane normal and cell normal
  My_Type::Vector3f plane_normal;
  if (is_valid_cell) {
    My_Type::Vector3f cell_normal;
    cell_normal.x = cell_info_mat[cell_index].nx;
    cell_normal.y = cell_info_mat[cell_index].ny;
    cell_normal.z = cell_info_mat[cell_index].nz;
    plane_normal.x = hist_normal[0].nx;
    plane_normal.y = hist_normal[0].ny;
    plane_normal.z = hist_normal[0].nz;
    float inner_product = cell_normal.x * plane_normal.x +
                          cell_normal.y * plane_normal.y +
                          cell_normal.z * plane_normal.z;

    if (inner_product < MIN_CELL_NORMAL_INNER_PRODUCT_VALUE)
      is_valid_cell = false;
  }

  // Compute project distance
  if (is_valid_cell) {
    My_Type::Vector3f cell_position;
    cell_position.x = cell_info_mat[cell_index].x;
    cell_position.y = cell_info_mat[cell_index].y;
    cell_position.z = cell_info_mat[cell_index].z;
    float project_distance = -plane_normal.x * cell_position.x -
                             plane_normal.y * cell_position.y -
                             plane_normal.z * cell_position.z;

    // Gaussian weighted
    int dist_index = roundf(project_distance / MIN_PLANE_DISTANCE);
    if ((dist_index > (int)2) &&
        (dist_index < (int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE) - 2)) {
      // 0.0359	0.1093	0.2129	0.2659	0.2129	0.1093	0.0359
      atomicAdd(&distance_histogram[dist_index - 2], 0.0359);
      atomicAdd(&distance_histogram[dist_index - 1], 0.1093);
      atomicAdd(&distance_histogram[dist_index + 0], 0.2659);
      atomicAdd(&distance_histogram[dist_index + 1], 0.1093);
      atomicAdd(&distance_histogram[dist_index + 2], 0.0359);
    }
  }
}
// Cpp call CUDA
void histogram_prj_dist_CUDA(dim3 block_rect, dim3 thread_rect,
                             const Cell_info *cell_info_mat,
                             const Hist_normal *hist_normal,
                             float *prj_dist_hist) {
  histogram_prj_dist_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, hist_normal, prj_dist_hist);
}

// Find peaks on distance histogram for each direction
__global__ void find_prj_dist_peaks_KernelFunc(const float *prj_dist_hist,
                                               const Hist_normal *hist_normal,
                                               int *peak_index,
                                               Plane_info *current_planes) {
  bool is_valid_dist = true;
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  //
  float number_of_cells[3];
  number_of_cells[1] = prj_dist_hist[index];
  // if (number_of_cells[1] < MIN_CELLS_OF_PLANE * 0.2659)
  // is_valid_dist = false;
  if (number_of_cells[1] < MIN_AREA_OF_DIRECTION) is_valid_dist = false;

  //
  if (is_valid_dist) {
    // Read neighbor value
    number_of_cells[0] = prj_dist_hist[index - 1];
    number_of_cells[2] = prj_dist_hist[index + 1];

    // If it's a maximum. Than plane detected.
    if (number_of_cells[0] < number_of_cells[1] &&
        number_of_cells[2] < number_of_cells[1]) {
      int peak_index_now = atomicAdd(peak_index, 1);
      //
      current_planes[peak_index_now].nx = hist_normal[0].nx;
      current_planes[peak_index_now].ny = hist_normal[0].ny;
      current_planes[peak_index_now].nz = hist_normal[0].nz;
      current_planes[peak_index_now].d = (float)index * MIN_PLANE_DISTANCE;
      current_planes[peak_index_now].weight = prj_dist_hist[index];
      current_planes[peak_index_now].is_valid = true;
    }
  }
}
// Cpp call CUDA
void find_prj_dist_peaks_CUDA(dim3 block_rect, dim3 thread_rect,
                              const float *prj_dist_hist,
                              const Hist_normal *hist_normal, int *peak_index,
                              Plane_info *current_planes) {
  find_prj_dist_peaks_KernelFunc<<<block_rect, thread_rect>>>(
      prj_dist_hist, hist_normal, peak_index, current_planes);
}

#pragma region(GPU K - means)
// 1. Mark plane label for each cells
__global__ void mark_plane_label_for_cells_KernelFunc(
    const Plane_info *current_plane, Cell_info *cell_info_mat, int plane_num) {
  int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
  bool is_valid_cell = true;

  // Validate cell
  int valid_counter = cell_info_mat[cell_index].counter;
  if (valid_counter < MIN_VALID_POINTS_IN_CELL) is_valid_cell = false;

  // Compute inlcuded angle
  if (is_valid_cell) {
    float plane_nx, plane_ny, plane_nz, nx, ny, nz;
    nx = cell_info_mat[cell_index].nx;
    ny = cell_info_mat[cell_index].ny;
    nz = cell_info_mat[cell_index].nz;
    //
    float px, py, pz;
    px = cell_info_mat[cell_index].x;
    py = cell_info_mat[cell_index].y;
    pz = cell_info_mat[cell_index].z;

    //
    int match_plane_index = 0;
    float min_para_distance = FLT_MAX;
    for (int i = 1; i < plane_num; i++) {
      if (current_plane[i].is_valid == false) continue;

      plane_nx = current_plane[i].nx;
      plane_ny = current_plane[i].ny;
      plane_nz = current_plane[i].nz;
      float inner_product = nx * plane_nx + ny * plane_ny + nz * plane_nz;
      if (inner_product <
          1.0f - 3 * (1.0f - MIN_CELL_NORMAL_INNER_PRODUCT_VALUE))
        continue;

      // Plane distance to origin
      float plane_d = current_plane[i].d;
      // Project distance
      float prj_d = -px * plane_nx - py * plane_ny - pz * plane_nz;
      // Different of distance
      float d_diff = fabsf(plane_d - prj_d);

      // 3 sigma criterion
      if (d_diff > 2 * MIN_PLANE_DISTANCE) continue;

      //
      float distance_value = (1 - inner_product) + d_diff;
      if (distance_value < min_para_distance) {
        min_para_distance = distance_value;
        match_plane_index = i;
      }
    }

    // Match to current planes
    cell_info_mat[cell_index].plane_index = match_plane_index;
  } else {
    // No matched plane
    cell_info_mat[cell_index].plane_index = 0;
  }
}
// 2. Reset mean value
__global__ void reset_K_means_state_KernelFunc(Plane_info *current_plane) {
  int plane_index = blockIdx.x;
  current_plane[plane_index].cell_num = 0.0;
  current_plane[plane_index].d = 0.0;
  current_plane[plane_index].area = 0.0;
}
// 3. Sum plane parameters
__global__ void compute_mean_plane_para_KernelFunc(
    const Cell_info *cell_info_mat, Cell_info *plane_mean_paramters,
    int plane_num) {
  int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
  bool is_valid_cell = true;

  // Validate cell information
  int cell_pixel_number = cell_info_mat[cell_index].counter;
  if (cell_pixel_number < MIN_VALID_POINTS_IN_CELL) is_valid_cell = false;
  //
  int cell_plane_index = cell_info_mat[cell_index].plane_index;
  if (cell_plane_index >= MAX_CURRENT_PLANES || cell_plane_index == 0)
    is_valid_cell = false;

  //
  float px, py, pz, cell_nx, cell_ny, cell_nz;
  if (is_valid_cell) {
    // Load cell position
    px = cell_info_mat[cell_index].x;
    py = cell_info_mat[cell_index].y;
    pz = cell_info_mat[cell_index].z;
  }

  //
  for (int plane_index = 1; plane_index < plane_num; plane_index++) {
    __shared__ float cache_f0[256], cache_f1[256], cache_f2[256], cache_f3[256];
    int tid = threadIdx.x;

#pragma region(Reduce cell position and number - of - cell)
    //
    if (is_valid_cell && plane_index == cell_plane_index) {
      cache_f0[tid] = px;
      cache_f1[tid] = py;
      cache_f2[tid] = pz;
      cache_f3[tid] = 1.0;
    } else {
      cache_f0[tid] = 0.0;
      cache_f1[tid] = 0.0;
      cache_f2[tid] = 0.0;
      cache_f3[tid] = 0.0;
    }
    // Reduce
    block_256_reduce(cache_f0, tid);
    block_256_reduce(cache_f1, tid);
    block_256_reduce(cache_f2, tid);
    block_256_reduce(cache_f3, tid);
    // Save reduce result
    if (tid == 0) {
      atomicAdd(&plane_mean_paramters[plane_index].x, cache_f0[0]);
      atomicAdd(&plane_mean_paramters[plane_index].y, cache_f1[0]);
      atomicAdd(&plane_mean_paramters[plane_index].z, cache_f2[0]);
      atomicAdd(&plane_mean_paramters[plane_index].counter, (int)cache_f3[0]);
    }
#pragma endregion

#pragma region(Reduce cell normal)
    //
    if (is_valid_cell && plane_index == cell_plane_index) {
      cache_f0[tid] = cell_nx;
      cache_f1[tid] = cell_ny;
      cache_f2[tid] = cell_nz;
    } else {
      cache_f0[tid] = 0.0;
      cache_f1[tid] = 0.0;
      cache_f2[tid] = 0.0;
    }
    // Reduce
    block_256_reduce(cache_f0, tid);
    block_256_reduce(cache_f1, tid);
    block_256_reduce(cache_f2, tid);
    //
    if (tid == 0) {
      atomicAdd(&plane_mean_paramters[plane_index].nx, cache_f0[0]);
      atomicAdd(&plane_mean_paramters[plane_index].ny, cache_f1[0]);
      atomicAdd(&plane_mean_paramters[plane_index].nz, cache_f2[0]);
    }
#pragma endregion

#pragma region(plane area)
    //
    if (is_valid_cell && plane_index == cell_plane_index) {
      cache_f0[tid] = cell_info_mat[cell_index].area;
    } else {
      cache_f0[tid] = 0.0;
    }
    //
    block_256_reduce(cache_f0, tid);
    //
    if (tid == 0)
      atomicAdd(&plane_mean_paramters[plane_index].area, cache_f0[0]);

#pragma endregion
  }
}
// 3. Compute mean parameter of planes
__global__ void compute_plane_mean_para_KernelFunc(
    Cell_info *plane_mean_paramters, int plane_num) {
  int plane_index = blockIdx.x;
  // Validate
  int cell_pixel_number = plane_mean_paramters[plane_index].counter;
  if (cell_pixel_number == 0) return;

  // Compute mean value
  plane_mean_paramters[plane_index].x /= (float)cell_pixel_number;
  plane_mean_paramters[plane_index].y /= (float)cell_pixel_number;
  plane_mean_paramters[plane_index].z /= (float)cell_pixel_number;
  plane_mean_paramters[plane_index].nx /= (float)cell_pixel_number;
  plane_mean_paramters[plane_index].ny /= (float)cell_pixel_number;
  plane_mean_paramters[plane_index].nz /= (float)cell_pixel_number;
}
// 4. Build local coordinate
__global__ void build_local_plane_coordinate_KernelFunc(
    const Plane_info *current_plane, Plane_coordinate *local_coordinate) {
  int plane_id = blockIdx.x + 1;

  Plane_coordinate temp_coordinate;
  temp_coordinate.z_vec.x = current_plane[plane_id].nx;
  temp_coordinate.z_vec.y = current_plane[plane_id].ny;
  temp_coordinate.z_vec.z = current_plane[plane_id].nz;

  generate_local_coordinate(temp_coordinate);
  local_coordinate[plane_id] = temp_coordinate;
}
// 4. Re-compute plane normal vector (step 1)
__global__ void recompute_plane_normal_step1_KernelFunc(
    const Cell_info *cell_info_mat, const Cell_info *plane_mean_paramters,
    float *ATA_upper_buffer, float *ATb_buffer,
    Plane_coordinate *local_coordinate, Plane_info *current_plane,
    int plane_num) {
  int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
  bool is_valid_cell = true;
  // Validate cell
  int cell_pixel_number = cell_info_mat[cell_index].counter;
  if (cell_pixel_number < MIN_VALID_POINTS_IN_CELL) is_valid_cell = false;
  // Validate plane index
  int cell_plane_index = cell_info_mat[cell_index].plane_index;
  if (cell_plane_index >= MAX_CURRENT_PLANES || cell_plane_index == 0)
    is_valid_cell = false;

  //
  for (int plane_index = 1; plane_index < plane_num; plane_index++) {
    My_Type::Vector3f cell_point, local_cell_point;
    if (is_valid_cell && plane_index == cell_plane_index) {
      // Load position
      cell_point.x =
          cell_info_mat[cell_index].x - plane_mean_paramters[plane_index].x;
      cell_point.y =
          cell_info_mat[cell_index].y - plane_mean_paramters[plane_index].y;
      cell_point.z =
          cell_info_mat[cell_index].z - plane_mean_paramters[plane_index].z;
      transform_to_local_coordinate(cell_point, local_cell_point,
                                    local_coordinate[plane_index]);
    }

    //
    __shared__ float cache_f[256];
    int tid = threadIdx.x;

    // ATA = {a0, a1}	|	ATb = {b0}
    //		 {a1, a2}	|		  {b1}
    float ATA_upper[3], ATb[2];
    // Reduce ATA and ATb
    if (is_valid_cell && plane_index == cell_plane_index) {
      ATA_upper[0] = local_cell_point.x * local_cell_point.x;
      ATA_upper[1] = local_cell_point.x * local_cell_point.y;
      ATA_upper[2] = local_cell_point.y * local_cell_point.y;
      ATb[0] = -local_cell_point.x * local_cell_point.z;
      ATb[1] = -local_cell_point.y * local_cell_point.z;
    } else {
      ATA_upper[0] = 0.0f;
      ATA_upper[1] = 0.0f;
      ATA_upper[2] = 0.0f;
      ATb[0] = 0.0f;
      ATb[1] = 0.0f;
    }
    // Reduce Hessian
    for (int i = 0; i < 3; i++) {
      __syncthreads();
      cache_f[tid] = ATA_upper[i];
      // Reduce
      block_256_reduce(cache_f, tid);
      if (tid == 0)
        atomicAdd(&ATA_upper_buffer[plane_index * 3 + i], cache_f[0]);
    }
    // Reduce Nabla
    for (int i = 0; i < 2; i++) {
      __syncthreads();
      cache_f[tid] = ATb[i];
      block_256_reduce(cache_f, tid);
      if (tid == 0) atomicAdd(&ATb_buffer[plane_index * 2 + i], cache_f[0]);
    }

    // Count valid cells
    if (is_valid_cell && plane_index == cell_plane_index) {
      cache_f[tid] = 1.0;
    } else {
      cache_f[tid] = 0.0;
    }
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&current_plane[plane_index].cell_num, cache_f[0]);
    if (tid == 0)
      current_plane[plane_index].area = plane_mean_paramters[plane_index].area;
  }
}
// 4. Re-compute plane normal vector (step 2)
__global__ void recompute_plane_normal_step2_KernelFunc(
    Plane_info *current_plane, Cell_info *plane_mean_paramters,
    float *ATA_upper_buffer, float *ATb_buffer,
    Plane_coordinate *local_coordinate) {
  int plane_index = blockIdx.x;
  // Validate
  if (!current_plane[plane_index].is_valid) return;

  //
  float ATA_upper[3], ATb[2];
  ATA_upper[0] = ATA_upper_buffer[plane_index * 3 + 0];
  ATA_upper[1] = ATA_upper_buffer[plane_index * 3 + 1];
  ATA_upper[2] = ATA_upper_buffer[plane_index * 3 + 2];
  ATb[0] = ATb_buffer[plane_index * 2 + 0];
  ATb[1] = ATb_buffer[plane_index * 2 + 1];

  // Cramer solver
  float D, D1, D2, tan_xz, tan_yz;
  D = ATA_upper[0] * ATA_upper[2] - ATA_upper[1] * ATA_upper[1];
  D1 = ATb[0] * ATA_upper[2] - ATb[1] * ATA_upper[1];
  D2 = ATb[1] * ATA_upper[0] - ATb[0] * ATA_upper[1];
  // compte tangent theta
  tan_xz = D1 / D;
  tan_yz = D2 / D;

  //
  My_Type::Vector3f local_normal_vec, normal_vec;
  local_normal_vec.z = 1 / norm3df(tan_xz, tan_yz, 1.0f);
  local_normal_vec.x = tan_xz * local_normal_vec.z;
  local_normal_vec.y = tan_yz * local_normal_vec.z;
  transform_from_local_coordinate(local_normal_vec, normal_vec,
                                  local_coordinate[plane_index]);

  // Check direction of normal vector (normal vector are difined as the vector
  // point to camera)
  My_Type::Vector3f ray_vec;
  ray_vec.x = plane_mean_paramters[plane_index].x;
  ray_vec.y = plane_mean_paramters[plane_index].y;
  ray_vec.z = plane_mean_paramters[plane_index].z;
  if ((ray_vec.dot(normal_vec)) > 0.0f) {
    normal_vec.x = -normal_vec.x;
    normal_vec.y = -normal_vec.y;
    normal_vec.z = -normal_vec.z;
  }

  // Update current plane direction
  normal_vec.normlize();
  current_plane[plane_index].nx = normal_vec.x;
  current_plane[plane_index].ny = normal_vec.y;
  current_plane[plane_index].nz = normal_vec.z;
}
// 5. Re-compute plane distance to origin
__global__ void recompute_plane_distance_step1_KernelFunc(
    const Cell_info *cell_info_mat, Plane_info *current_plane, int plane_num) {
  int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
  bool is_valid_cell = true;

  // Validate cell
  int valid_counter = cell_info_mat[cell_index].counter;
  if (valid_counter < MIN_VALID_POINTS_IN_CELL) is_valid_cell = false;
  // Validate plane
  int cell_plane_index = cell_info_mat[cell_index].plane_index;
  if (cell_plane_index > MAX_CURRENT_PLANES || cell_plane_index == 0)
    is_valid_cell = false;

  //
  __shared__ float cache_f[256];
  int tid = threadIdx.x;
  //
  for (int plane_index = 1; plane_index < plane_num; plane_index++) {
    float plane_nx, plane_ny, plane_nz, prj_distance;
    // Load plane normal direction
    plane_nx = current_plane[plane_index].nx;
    plane_ny = current_plane[plane_index].ny;
    plane_nz = current_plane[plane_index].nz;
    // Compute projection distance to origin
    if (is_valid_cell && cell_plane_index == plane_index) {
      float cell_x = cell_info_mat[cell_index].x;
      float cell_y = cell_info_mat[cell_index].y;
      float cell_z = cell_info_mat[cell_index].z;

      prj_distance = -plane_nx * cell_x - plane_ny * cell_y - plane_nz * cell_z;
    } else {
      prj_distance = 0.0;
    }

    // Reduce plane distance
    cache_f[tid] = prj_distance;
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&current_plane[plane_index].d, cache_f[0]);
  }
}
// ��2��
__global__ void recompute_plane_distance_step2_KernelFunc(
    Plane_info *current_plane) {
  //
  int plane_index = blockIdx.x;
  current_plane[plane_index].d /= current_plane[plane_index].cell_num;
  // Validate
  // if (current_plane[plane_index].area < MIN_AREA_OF_DIRECTION * 0.2)
  if (current_plane[plane_index].cell_num < 10) {
    current_plane[plane_index].is_valid = false;
  } else {
    current_plane[plane_index].is_valid = true;
  }
}
//
__global__ void merge_similar_planes_KernelFunc(Plane_info *current_plane,
                                                int plane_number) {
  const float normal_threshold = 0.98f;
  const float distance_threshold = 0.04f;

  for (int plane_id = 1; plane_id < plane_number; plane_id++) {
    if (!current_plane[plane_id].is_valid) continue;

    My_Type::Vector3f plane_normal;
    plane_normal.x = current_plane[plane_id].nx;
    plane_normal.y = current_plane[plane_id].ny;
    plane_normal.z = current_plane[plane_id].nz;
    float plane_distance = current_plane[plane_id].d;

    //
    for (int check_id = plane_id + 1; check_id < plane_number; check_id++) {
      My_Type::Vector3f check_normal;
      check_normal.x = current_plane[check_id].nx;
      check_normal.y = current_plane[check_id].ny;
      check_normal.z = current_plane[check_id].nz;
      float check_distance = current_plane[check_id].d;

      if (plane_normal.dot(check_normal) > normal_threshold &&
          fabsf(plane_distance - check_distance) < distance_threshold) {
        current_plane[check_id].is_valid = false;
      }
    }
  }
}
// Cpp call CUDA
void K_mean_iterate_CUDA(dim3 block_rect, dim3 thread_rect,
                         Plane_info *current_plane, Cell_info *cell_info_mat,
                         Cell_info *plane_mean_paramters,
                         Plane_coordinate *local_coordinate,
                         float *ATA_upper_buffer, float *ATb_buffer,
                         int plane_num) {
  // 1. Mark plane label for each cell
  mark_plane_label_for_cells_KernelFunc<<<block_rect, thread_rect>>>(
      current_plane, cell_info_mat, plane_num);

  // 2. Reset plane parameters
  reset_K_means_state_KernelFunc<<<plane_num, 1>>>(current_plane);

  // 3. Compute mean value of each planes (plane label are marked in step 1.)
  compute_mean_plane_para_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, plane_mean_paramters, plane_num);
  compute_plane_mean_para_KernelFunc<<<plane_num, 1>>>(plane_mean_paramters,
                                                       plane_num);

  // 4. Recompute current plane direction
  // Build local coordinate
  build_local_plane_coordinate_KernelFunc<<<plane_num, 1>>>(current_plane,
                                                            local_coordinate);
  //
  recompute_plane_normal_step1_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, plane_mean_paramters, ATA_upper_buffer, ATb_buffer,
      local_coordinate, current_plane, plane_num);
  recompute_plane_normal_step2_KernelFunc<<<plane_num, 1>>>(
      current_plane, plane_mean_paramters, ATA_upper_buffer, ATb_buffer,
      local_coordinate);

  // 5. Recompute current plane distance to origin
  // printf("block_rect : %d, %d, %d\n", block_rect.x, block_rect.y,
  // block_rect.z);
  recompute_plane_distance_step1_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, current_plane, plane_num);
  recompute_plane_distance_step2_KernelFunc<<<plane_num, 1>>>(current_plane);

  // 6. Merge similar planes
  merge_similar_planes_KernelFunc<<<1, 1>>>(current_plane, plane_num);
}
#pragma endregion

//
__global__ void label_current_planes_KernelFunc(const Cell_info *cell_info_mat,
                                                int *current_plane_labels) {
  // Coordinate/index of cell
  int u_cell = blockIdx.x;
  int v_cell = blockIdx.y;
  int cell_index = u_cell + v_cell * gridDim.x;
  // Coordinate/index of pixel
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;
  int index = u + v * gridDim.x * blockDim.x;

  //
  int plane_label = cell_info_mat[cell_index].plane_index;
  current_plane_labels[index] = plane_label;
}
//
void label_current_planes_CUDA(dim3 block_rect, dim3 thread_rect,
                               const Cell_info *cell_info_mat,
                               int *current_plane_labels) {
  label_current_planes_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, current_plane_labels);
}

//
__global__ void relabel_plane_labels_KernelFunc(
    const My_Type::Vector2i *matches, int *current_plane_labels) {
  // Coordinate/index of pixel
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;
  int index = u + v * gridDim.x * blockDim.x;

  //
  int current_label = current_plane_labels[index];
  current_plane_labels[index] = matches[current_label].y;
}
//
void relabel_plane_labels_CUDA(dim3 block_rect, dim3 thread_rect,
                               const My_Type::Vector2i *matches,
                               int *current_plane_labels) {
  relabel_plane_labels_KernelFunc<<<block_rect, thread_rect>>>(
      matches, current_plane_labels);
}

//
__global__ void count_planar_pixel_number_KernelFunc(const int *plane_labels,
                                                     Plane_info *plane_list,
                                                     int plane_counter) {
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  int index = u + v * gridDim.x * blockDim.x;

  int plane_label = plane_labels[index];
  //
  __shared__ int counter_cache[256];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  // Count each plane
  for (int plane_index = 0; plane_index < plane_counter; plane_index++) {
    //
    if (plane_index == plane_label) {
      counter_cache[tid] = 1;
    } else {
      counter_cache[tid] = 0;
    }

    //
    block_256_reduce(counter_cache, tid);
    if (tid == 0) {
      atomicAdd(&plane_list[plane_index].pixel_num, counter_cache[0]);
    }
  }
}
//
void count_planar_pixel_number_CUDA(dim3 block_rect, dim3 thread_rect,
                                    const int *plane_labels,
                                    Plane_info *plane_list, int plane_counter) {
  count_planar_pixel_number_KernelFunc<<<block_rect, thread_rect>>>(
      plane_labels, plane_list, plane_counter);
}

//
__global__ void count_overlap_pixel_number_KernelFunc(
    const int *current_plane_labels, const int *model_plane_labels,
    int current_plane_counter, int *relative_matrix) {
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  int index = u + v * gridDim.x * blockDim.x;

  //
  int current_label = current_plane_labels[index];
  int model_label = model_plane_labels[index];

  //
  __shared__ int model_plane_counter_cache[MAX_MODEL_PLANES];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  for (int plane_id = 0; plane_id < current_plane_counter; plane_id++) {
    for (int i = 0; i < MAX_MODEL_PLANES / 256; i++) {
      model_plane_counter_cache[tid + 256 * i] = 0;
    }
    __syncthreads();

    if (plane_id == current_label) {
      atomicAdd(&model_plane_counter_cache[model_label], 1);
    }
    __syncthreads();

    //
    for (int i = 0; i < MAX_MODEL_PLANES / 256; i++) {
      atomicAdd(&relative_matrix[(tid + 256 * i) + MAX_MODEL_PLANES * plane_id],
                model_plane_counter_cache[(tid + 256 * i)]);
    }
    __syncthreads();
  }
}
//
void count_overlap_pixel_number_CUDA(dim3 block_rect, dim3 thread_rect,
                                     const int *current_plane_labels,
                                     const int *model_plane_labels,
                                     int current_plane_counter,
                                     int *relative_matrix) {
  count_overlap_pixel_number_KernelFunc<<<block_rect, thread_rect>>>(
      current_plane_labels, model_plane_labels, current_plane_counter,
      relative_matrix);
}

// ------------------------------------ Super Pixel Functions :
#define INVALID_PXIEL -1
#define OUTLAYER_OF_SUPER_PIXEL -2
//
__global__ void init_super_pixel_image_KernelFunc(
    const My_Type::Vector3f *points, const My_Type::Vector3f *normals,
    int *super_pixel_id_image, int super_pixel_width,
    int number_of_block_per_line) {
  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;
  if (offset_x >= super_pixel_width || offset_y >= super_pixel_width) return;

  // compute the start of the search window
  int start_x = blockIdx.x * super_pixel_width;
  int start_y = blockIdx.y * super_pixel_width;
  //
  int u = start_x + offset_x;
  int v = start_y + offset_y;
  //
  int image_width = blockDim.x * gridDim.x;
  int image_height = blockDim.y * gridDim.y;
  int pixel_index = u + v * image_width;
  if (u < 0 || u >= image_width || v < 0 || v >= image_height) return;

  // Validate point position and normal vector
  bool is_valid_pixel;
  My_Type::Vector3f normal_vec(0.0f), point_vec(0.0f);
  normal_vec = normals[pixel_index];
  point_vec = points[pixel_index];
  if (point_vec.z > FLT_EPSILON && normal_vec.norm() > FLT_EPSILON) {
    is_valid_pixel = true;
  } else {
    is_valid_pixel = false;
  }

  // -1 : Invalid pixel id
  int super_pixel_id = INVALID_PXIEL;
  if (is_valid_pixel) super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  super_pixel_id_image[pixel_index] = super_pixel_id;
}
//
void init_super_pixel_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                 const My_Type::Vector3f *points,
                                 const My_Type::Vector3f *normals,
                                 int *super_pixel_id_image,
                                 int super_pixel_width,
                                 int number_of_block_per_line) {
  init_super_pixel_image_KernelFunc<<<block_rect, thread_rect>>>(
      points, normals, super_pixel_id_image, super_pixel_width,
      number_of_block_per_line);
}

//
__global__ void update_cluster_center_KernelFunc(
    const My_Type::Vector3f *points, const My_Type::Vector3f *normals,
    const int *super_pixel_id_image, Super_pixel *accumulate_super_pixels,
    int super_pixel_width, int number_of_block_per_line) {
  // Important! this reduce many usefuless atomic operations (speed up 1.5x)
  __shared__ bool block_need_add;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_need_add = false;
  //
  bool point_need_add = false;
  My_Type::Vector2f center_vec(0.0f);
  My_Type::Vector3f normal_vec(0.0f), point_vec(0.0f);

  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  if (offset_x < super_pixel_width * 3 && offset_y < super_pixel_width * 3) {
    // compute the start of the search window
    int start_x = blockIdx.x * super_pixel_width - super_pixel_width;
    int start_y = blockIdx.y * super_pixel_width - super_pixel_width;
    //
    int u = start_x + offset_x;
    int v = start_y + offset_y;
    //
    int image_width = blockDim.x * gridDim.x;
    int image_height = blockDim.y * gridDim.y;
    int pixel_index = u + v * image_width;

    if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
      if (super_pixel_id_image[pixel_index] == super_pixel_id) {
        point_vec = points[pixel_index];
        normal_vec = normals[pixel_index];
        center_vec.u = u;
        center_vec.v = v;
        point_need_add = true;
        block_need_add = true;
      }
    }
  }
  // No point need add in this block
  __syncthreads();
  if (!block_need_add) return;

  __shared__ float cache_f[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  __shared__ int cache_i[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  // Reduce normals
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = normal_vec.data[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    // Reduction add
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].normal_data[i],
                cache_f[0]);
  }
  // Reduce position
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = point_vec.data[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    // Reduction add
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].position_data[i],
                cache_f[0]);
  }
  // Reduce center
  for (int i = 0; i < 2; i++) {
    if (point_need_add) {
      cache_f[tid] = center_vec.data[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    // Reduction add
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0)
      atomicAdd(
          (float *)&accumulate_super_pixels[super_pixel_id].center_data[i],
          cache_f[0]);
  }
  __syncthreads();
  // Reduce valid counter
  {
    if (point_need_add) {
      cache_i[tid] = 1;
    } else {
      cache_i[tid] = 0;
    }
    // Reduction add
    block_256_reduce(cache_i, tid);
    //
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].valid_pixel_number,
                cache_i[0]);
  }
}
//
__global__ void process_accumulate_super_pixel_KernelFunc(
    const Super_pixel *accumulate_super_pixels, int number_of_block_per_line,
    Super_pixel *super_pixels) {
  //
  My_Type::Vector2f center_vec(0.0f);
  My_Type::Vector3f normal_vec(0.0f), point_vec(0.0f);
  Super_pixel temp_sp;
  int valid_counter = 0;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  //
  temp_sp = accumulate_super_pixels[super_pixel_id];
  center_vec.x += temp_sp.cx;
  center_vec.y += temp_sp.cy;
  normal_vec.x += temp_sp.nx;
  normal_vec.y += temp_sp.ny;
  normal_vec.z += temp_sp.nz;
  point_vec.x += temp_sp.px;
  point_vec.y += temp_sp.py;
  point_vec.z += temp_sp.pz;
  valid_counter += temp_sp.valid_pixel_number;

  //
  if (valid_counter) {
    center_vec /= valid_counter;
    normal_vec /= valid_counter;
    point_vec /= valid_counter;
    normal_vec.normlize();
  }

  //
  temp_sp.cx = center_vec.x;
  temp_sp.cy = center_vec.y;
  temp_sp.nx = normal_vec.x;
  temp_sp.ny = normal_vec.y;
  temp_sp.nz = normal_vec.z;
  temp_sp.px = point_vec.x;
  temp_sp.py = point_vec.y;
  temp_sp.pz = point_vec.z;
  temp_sp.valid_pixel_number = valid_counter;

  int super_pixel_index = blockIdx.x + gridDim.x * blockIdx.y;
  super_pixels[super_pixel_index] = temp_sp;
}
//
void update_cluster_center_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *points,
    const My_Type::Vector3f *normals, const int *super_pixel_id_image,
    Super_pixel *accumulate_super_pixels, Super_pixel *super_pixels,
    int super_pixel_width, int number_of_block_per_line) {
  update_cluster_center_KernelFunc<<<block_rect, thread_rect>>>(
      points, normals, super_pixel_id_image, accumulate_super_pixels,
      super_pixel_width, number_of_block_per_line);

  //
  dim3 block_rect_2 = block_rect;
  dim3 thread_rect_2(1, 1, 1);
  block_rect_2.z = 1;
  process_accumulate_super_pixel_KernelFunc<<<block_rect_2, thread_rect_2>>>(
      accumulate_super_pixels, number_of_block_per_line, super_pixels);
}

//
__device__ inline float compute_super_pixel_distance(
    const Super_pixel &temp_sp, const My_Type::Vector3f &normal_vec,
    const My_Type::Vector3f &point_vec, const My_Type::Vector2f &center_vec,
    float &weight_pixel_data, float &weight_normal_position,
    Sensor_params sensor_params, int &super_pixel_width) {
  My_Type::Vector3f sp_normal(temp_sp.nx, temp_sp.ny, temp_sp.nz);
  My_Type::Vector3f sp_point(temp_sp.px, temp_sp.py, temp_sp.pz);
  My_Type::Vector2f sp_center(temp_sp.cx, temp_sp.cy);

  My_Type::Vector3f diff_vec = sp_point - point_vec;
  float point_distance = diff_vec.norm();
  float outlayer_distance =
      sp_point.z * super_pixel_width * 3.0 / sensor_params.sensor_fx;
  if (point_distance > outlayer_distance) return FLT_MAX;
  // if (point_distance > 0.15)		return FLT_MAX;

  //
  float distance_normal = 1 - sp_normal.dot(normal_vec);
  //
  // float distance_point = point_distance / (sensor_params.sensor_noise_ratio *
  // point_vec.z);
  float distance_laplacian = fabsf(diff_vec.dot(sp_normal)) /
                             (sensor_params.sensor_noise_ratio * point_vec.z);
  //
  float distance_center = (sp_center - center_vec).norm();
  distance_center /= (float)super_pixel_width;
  //
  distance_center *= weight_pixel_data;
  distance_normal *= (1.0f - weight_pixel_data) * weight_normal_position;
  distance_laplacian *=
      (1.0f - weight_pixel_data) * (1.0f - weight_normal_position);

  return (distance_normal + distance_laplacian + distance_center);
}
//
__global__ void pixel_find_associate_center_KernelFunc(
    const My_Type::Vector3f *points, const My_Type::Vector3f *normals,
    const Super_pixel *super_pixels, int *super_pixel_id_image,
    int super_pixel_width, float weight_pixel_data,
    float weight_normal_position, Sensor_params sensor_params) {
  //
  bool is_valid_pixel = true;

  //
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  int image_width = gridDim.x * blockDim.x;
  int pixel_index = u + v * image_width;
  if (super_pixel_id_image[pixel_index] == INVALID_PXIEL)
    is_valid_pixel = false;

  //
  int min_super_pixel_id = OUTLAYER_OF_SUPER_PIXEL;
  if (is_valid_pixel) {
    //
    My_Type::Vector3f normal_vec(0.0f), point_vec(0.0f);
    My_Type::Vector2f center_vec((float)u, (float)v);
    normal_vec = normals[pixel_index];
    point_vec = points[pixel_index];

    //
    int u_sp = u / super_pixel_width;
    int v_sp = v / super_pixel_width;
    //
    float min_distance = FLT_MAX;
    for (int v_offset = -1; v_offset <= 1; v_offset++)
      for (int u_offset = -1; u_offset <= 1; u_offset++) {
        int u_sp_check = u_sp + u_offset;
        int v_sp_check = v_sp + v_offset;
        //
        int super_pixel_mat_width = gridDim.x;
        int super_pixel_mat_height = gridDim.y;
        //
        if (u_sp_check >= 0 && u_sp_check < super_pixel_mat_width &&
            v_sp_check >= 0 && v_sp_check < super_pixel_mat_height) {
          int super_pixel_index = u_sp_check + v_sp_check * gridDim.x;
          Super_pixel sp_check = super_pixels[super_pixel_index];

          // Compute distance to cluster center
          float distance = compute_super_pixel_distance(
              sp_check, normal_vec, point_vec, center_vec, weight_pixel_data,
              weight_normal_position, sensor_params, super_pixel_width);
          //
          if (distance < min_distance) {
            min_distance = distance;
            min_super_pixel_id = super_pixel_index;
          }
        }
      }
  }

  //
  super_pixel_id_image[pixel_index] = min_super_pixel_id;
}
//
void pixel_find_associate_center_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *points,
    const My_Type::Vector3f *normals, const Super_pixel *super_pixels,
    int *super_pixel_id_image, int super_pixel_width, float weight_pixel_data,
    float weight_normal_position, Sensor_params sensor_params) {
  pixel_find_associate_center_KernelFunc<<<block_rect, thread_rect>>>(
      points, normals, super_pixels, super_pixel_id_image, super_pixel_width,
      weight_pixel_data, weight_normal_position, sensor_params);
}

//
__device__ void generate_local_coordinate(Plane_coordinate &local_coordinate) {
  local_coordinate.x_vec = My_Type::Vector3f(1.0f, 0.0f, 0.0f);
  if (fabsf(local_coordinate.x_vec.dot(local_coordinate.z_vec)) >= 0.71) {
    local_coordinate.x_vec = My_Type::Vector3f(0.0f, 1.0f, 0.0f);
    if (fabsf(local_coordinate.x_vec.dot(local_coordinate.z_vec)) >= 0.71) {
      local_coordinate.x_vec = My_Type::Vector3f(0.0f, 0.0f, 1.0f);
    }
  }
  // Orthogonalization
  local_coordinate.x_vec = local_coordinate.x_vec -
                           local_coordinate.x_vec.dot(local_coordinate.z_vec) *
                               local_coordinate.z_vec;
  // Normalize base_x
  local_coordinate.x_vec.normlize();
  //
  local_coordinate.y_vec = local_coordinate.z_vec.cross(local_coordinate.x_vec);
}
//
__device__ inline void transform_to_local_coordinate(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate) {
  dst_point.x = local_coordinate.x_vec.dot(src_point);
  dst_point.y = local_coordinate.y_vec.dot(src_point);
  dst_point.z = local_coordinate.z_vec.dot(src_point);
}
//
__device__ inline void transform_from_local_coordinate(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate) {
  dst_point = src_point.x * local_coordinate.x_vec +
              src_point.y * local_coordinate.y_vec +
              src_point.z * local_coordinate.z_vec;
}
//
__global__ void generate_base_vector_for_each_cell_KernelFunc(
    const Super_pixel *super_pixels, Plane_coordinate *base_vectors) {
  // Coordinate/index of cell
  int u_super_pixel = blockIdx.x;
  int v_super_pixel = blockIdx.y;
  int super_pixel_index = u_super_pixel + v_super_pixel * gridDim.x;

  // Load super pixel initial value
  Super_pixel temp_sp = super_pixels[super_pixel_index];
  Plane_coordinate temp_coordinate;
  temp_coordinate.z_vec.x = temp_sp.nx;
  temp_coordinate.z_vec.y = temp_sp.ny;
  temp_coordinate.z_vec.z = temp_sp.nz;
  generate_local_coordinate(temp_coordinate);

  //
  base_vectors[super_pixel_index] = temp_coordinate;
}
//
#pragma region(ICP method)
//
__global__ void fit_plane_for_cells_KernelFunc(
    const My_Type::Vector3f *points, const int *super_pixel_id_image,
    const Plane_coordinate *base_vectors, const Super_pixel *super_pixels,
    Super_pixel *accumulate_super_pixels, My_Type::Vector3f *cell_hessain,
    My_Type::Vector2f *cell_nabla, int super_pixel_width,
    int number_of_block_per_line) {
  // Important! this reduce many usefuless atomic operations (speed up 1.5x)
  __shared__ bool block_need_add;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_need_add = false;
  //
  bool point_need_add = false;
  My_Type::Vector3f point_vec(0.0f);

  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  if (offset_x < super_pixel_width * 3 && offset_y < super_pixel_width * 3) {
    // compute the start of the search window
    int start_x = blockIdx.x * super_pixel_width - super_pixel_width;
    int start_y = blockIdx.y * super_pixel_width - super_pixel_width;
    //
    int u = start_x + offset_x;
    int v = start_y + offset_y;
    //
    int image_width = blockDim.x * gridDim.x;
    int image_height = blockDim.y * gridDim.y;
    int pixel_index = u + v * image_width;

    if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
      if (super_pixel_id_image[pixel_index] == super_pixel_id) {
        point_vec = points[pixel_index];
        point_need_add = true;
        block_need_add = true;
      }
    }
  }
  // No point need add in this block
  __syncthreads();
  if (!block_need_add) return;

  //
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  //
  __shared__ My_Type::Vector3f cell_normal_s, cell_position_s;
  if (tid == 0) {
    cell_normal_s.x = super_pixels[super_pixel_id].nx;
    cell_normal_s.y = super_pixels[super_pixel_id].ny;
    cell_normal_s.z = super_pixels[super_pixel_id].nz;
    cell_position_s.x = super_pixels[super_pixel_id].px;
    cell_position_s.y = super_pixels[super_pixel_id].py;
    cell_position_s.z = super_pixels[super_pixel_id].pz;
  }
  __syncthreads();

  // Compute jacobian, nabla and hessain
  Plane_coordinate cell_base_coordinate = base_vectors[super_pixel_id];
  float jacobian[2], residual, hessain_upper[3], nabla[2];
  if (point_need_add) {
    My_Type::Vector3f point_transfer = point_vec - cell_position_s;
    jacobian[0] = (point_transfer.dot(cell_base_coordinate.x_vec));
    jacobian[1] = (point_transfer.dot(cell_base_coordinate.y_vec));
    residual = fabsf(point_transfer.dot(cell_normal_s));
  } else {
    jacobian[0] = 0;
    jacobian[1] = 0;
    residual = 0;
  }
  hessain_upper[0] = jacobian[0] * jacobian[0];
  hessain_upper[1] = jacobian[0] * jacobian[1];
  hessain_upper[2] = jacobian[1] * jacobian[1];
  nabla[0] = jacobian[0] * residual;
  nabla[1] = jacobian[1] * residual;

  //
  __shared__ float cache_f[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  __shared__ int cache_i[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  // Reduce Hessain upper
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = hessain_upper[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    //
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&cell_hessain[super_pixel_id].data[i], cache_f[0]);
  }
  // Reduce nabla
  for (int i = 0; i < 2; i++) {
    if (point_need_add) {
      cache_f[tid] = nabla[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    //
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&cell_nabla[super_pixel_id].data[i], cache_f[0]);
  }
  // Reduce position
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = point_vec.data[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    // Reduction add
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].position_data[i],
                cache_f[0]);
  }
  // Reduce counter
  {
    if (point_need_add) {
      cache_i[tid] = 1;
    } else {
      cache_i[tid] = 0;
    }
    //
    block_256_reduce(cache_i, tid);
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].valid_pixel_number,
                cache_i[0]);
  }
}
//
__global__ void update_super_pixel_params_KenrelFunc(
    const Super_pixel *accumulate_super_pixels,
    const My_Type::Vector3f *cell_hessain_upper,
    const My_Type::Vector2f *cell_nabla,
    const Plane_coordinate *cell_base_vectors, Super_pixel *super_pixels) {
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;

  //
  Super_pixel temp_sp = accumulate_super_pixels[super_pixel_id];
  if (temp_sp.valid_pixel_number > 0) {
    // Position
    temp_sp.px /= (float)temp_sp.valid_pixel_number;
    temp_sp.py /= (float)temp_sp.valid_pixel_number;
    temp_sp.pz /= (float)temp_sp.valid_pixel_number;

    // Re-compute normal vector
    My_Type::Vector3f hessain_upper =
        cell_hessain_upper[super_pixel_id] / (float)temp_sp.valid_pixel_number;
    My_Type::Vector2f nabla =
        cell_nabla[super_pixel_id] / temp_sp.valid_pixel_number;
    float hessain_inv_upper[3], inc_coeffs[2];
    hessain_inv_upper[0] = 1 / hessain_upper.data[0];
    hessain_inv_upper[1] = -hessain_upper.data[1] /
                           (hessain_upper.data[0] * hessain_upper.data[2]);
    hessain_inv_upper[2] = 1 / hessain_upper.data[2];
    // G-N solver : X = inv(J'J) * J'r; nabla = J'r
    inc_coeffs[0] =
        hessain_inv_upper[0] * nabla[0] + hessain_inv_upper[1] * nabla[1];
    inc_coeffs[1] =
        hessain_inv_upper[1] * nabla[0] + hessain_inv_upper[2] * nabla[1];
    // Update normal vector
    Plane_coordinate base_coordinate = cell_base_vectors[super_pixel_id];
    My_Type::Vector3f new_normal;
    new_normal = inc_coeffs[0] * base_coordinate.x_vec +
                 inc_coeffs[1] * base_coordinate.y_vec + base_coordinate.z_vec;
    new_normal.normlize();
    //
    temp_sp.nx = new_normal.x;
    temp_sp.ny = new_normal.y;
    temp_sp.nz = new_normal.z;

    // for debug -----------------------------------------------
    temp_sp.is_planar_cell = true;
    // debug code end
  }

  // Update super pixel params
  super_pixels[super_pixel_id] = temp_sp;
}
#pragma endregion
//
#pragma region(tangent angle fit)
//
__global__ void fit_plane_for_cells_KernelFunc_2(
    const My_Type::Vector3f *points, const int *super_pixel_id_image,
    const Plane_coordinate *base_vectors, const Super_pixel *super_pixels,
    Super_pixel *accumulate_super_pixels, My_Type::Vector3f *cell_hessain,
    My_Type::Vector2f *cell_nabla, int super_pixel_width,
    int number_of_block_per_line) {
  // Important! this reduce many usefuless atomic operations (speed up 1.5x)
  __shared__ bool block_need_add;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_need_add = false;
  //
  bool point_need_add = false;
  My_Type::Vector3f point_vec(0.0f);

  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  if (offset_x < super_pixel_width * 3 && offset_y < super_pixel_width * 3) {
    // compute the start of the search window
    int start_x = blockIdx.x * super_pixel_width - super_pixel_width;
    int start_y = blockIdx.y * super_pixel_width - super_pixel_width;
    //
    int u = start_x + offset_x;
    int v = start_y + offset_y;
    //
    int image_width = blockDim.x * gridDim.x;
    int image_height = blockDim.y * gridDim.y;
    int pixel_index = u + v * image_width;

    if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
      if (super_pixel_id_image[pixel_index] == super_pixel_id) {
        point_vec = points[pixel_index];
        point_need_add = true;
        block_need_add = true;
      }
    }
  }
  // No point need add in this block
  __syncthreads();
  if (!block_need_add) return;

  //
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  //
  __shared__ My_Type::Vector3f cell_position_s;
  if (tid == 0) {
    cell_position_s.x = super_pixels[super_pixel_id].px;
    cell_position_s.y = super_pixels[super_pixel_id].py;
    cell_position_s.z = super_pixels[super_pixel_id].pz;
  }
  __syncthreads();

  // Compute jacobian, nabla and hessain
  float hessain_upper[3], nabla[2];
  if (point_need_add) {
    My_Type::Vector3f point_transfer = point_vec - cell_position_s;
    // Transfer to Super pixel coordinate
    Plane_coordinate cell_base_coordinate = base_vectors[super_pixel_id];
    My_Type::Vector3f point_local;
    transform_to_local_coordinate(point_transfer, point_local,
                                  cell_base_coordinate);

    //
    hessain_upper[0] = point_local.x * point_local.x;
    hessain_upper[1] = point_local.x * point_local.y;
    hessain_upper[2] = point_local.y * point_local.y;
    nabla[0] = -point_local.x * point_local.z;
    nabla[1] = -point_local.y * point_local.z;
  } else {
    hessain_upper[0] = 0;
    hessain_upper[1] = 0;
    hessain_upper[2] = 0;
    nabla[0] = 0;
    nabla[1] = 0;
  }

  //
  __shared__ float cache_f[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  __shared__ int cache_i[SUPER_PIXEL_BLOCK_WIDTH * SUPER_PIXEL_BLOCK_WIDTH];
  // Reduce Hessain upper
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = hessain_upper[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    //
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&cell_hessain[super_pixel_id].data[i], cache_f[0]);
  }
  // Reduce nabla
  for (int i = 0; i < 2; i++) {
    if (point_need_add) {
      cache_f[tid] = nabla[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    //
    block_256_reduce(cache_f, tid);
    if (tid == 0) atomicAdd(&cell_nabla[super_pixel_id].data[i], cache_f[0]);
  }
  // Reduce position
  for (int i = 0; i < 3; i++) {
    if (point_need_add) {
      cache_f[tid] = point_vec.data[i];
    } else {
      cache_f[tid] = 0.0f;
    }
    // Reduction add
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].position_data[i],
                cache_f[0]);
  }
  // Reduce counter
  {
    if (point_need_add) {
      cache_i[tid] = 1;
    } else {
      cache_i[tid] = 0;
    }
    //
    block_256_reduce(cache_i, tid);
    if (tid == 0)
      atomicAdd(&accumulate_super_pixels[super_pixel_id].valid_pixel_number,
                cache_i[0]);
  }
}
//
__global__ void update_super_pixel_params_KenrelFunc_2(
    const Super_pixel *accumulate_super_pixels,
    const My_Type::Vector3f *cell_hessain_upper,
    const My_Type::Vector2f *cell_nabla,
    const Plane_coordinate *cell_base_vectors, Super_pixel *super_pixels) {
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;

  //
  Super_pixel temp_sp = accumulate_super_pixels[super_pixel_id];
  if (temp_sp.valid_pixel_number > 0) {
    // Position
    temp_sp.px /= (float)temp_sp.valid_pixel_number;
    temp_sp.py /= (float)temp_sp.valid_pixel_number;
    temp_sp.pz /= (float)temp_sp.valid_pixel_number;

    // printf("%f, %f, %f\n", temp_sp.px, temp_sp.py, temp_sp.pz);

    // Re-compute normal vector
    My_Type::Vector3f hessain_upper =
        cell_hessain_upper[super_pixel_id] / (float)temp_sp.valid_pixel_number;
    My_Type::Vector2f nabla =
        cell_nabla[super_pixel_id] / temp_sp.valid_pixel_number;
    //
    float D, D1, D2, tan_xz, tan_yz;
    // Compute Crammer
    D = hessain_upper[0] * hessain_upper[2] -
        hessain_upper[1] * hessain_upper[1];
    D1 = nabla[0] * hessain_upper[2] - nabla[1] * hessain_upper[1];
    D2 = nabla[1] * hessain_upper[0] - nabla[0] * hessain_upper[1];
    // compute tangent
    tan_xz = D1 / D;
    tan_yz = D2 / D;
    //
    My_Type::Vector3f normal_vec_local, normal_vec;
    normal_vec_local.z = 1 / norm3df(tan_xz, tan_yz, 1.0f);
    normal_vec_local.x = tan_xz * normal_vec_local.z;
    normal_vec_local.y = tan_yz * normal_vec_local.z;
    // Update normal vector
    Plane_coordinate cell_local_coordinate = cell_base_vectors[super_pixel_id];
    transform_from_local_coordinate(normal_vec_local, normal_vec,
                                    cell_local_coordinate);
    // Check direction
    My_Type::Vector3f ray(temp_sp.px, temp_sp.py, temp_sp.pz);
    if (ray.dot(normal_vec) > 0.0f) {
      normal_vec.x = -normal_vec.x;
      normal_vec.y = -normal_vec.y;
      normal_vec.z = -normal_vec.z;
    }
    //
    temp_sp.nx = normal_vec.x;
    temp_sp.ny = normal_vec.y;
    temp_sp.nz = normal_vec.z;

    // for debug -----------------------------------------------
    temp_sp.is_planar_cell = true;
    // debug code end
  }

  // Update super pixel params
  super_pixels[super_pixel_id] = temp_sp;
}

#pragma endregion
//
void fit_plane_for_cells_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *points,
    const int *super_pixel_id_image, Plane_coordinate *base_vectors,
    Super_pixel *super_pixels, Super_pixel *accumulate_super_pixels,
    My_Type::Vector3f *cell_hessain_upper, My_Type::Vector2f *cell_nabla,
    int super_pixel_width, int number_of_block_per_line) {
  // Generate tangent plane base vectors
  generate_base_vector_for_each_cell_KernelFunc<<<block_rect, 1>>>(
      super_pixels, base_vectors);
  if (false) {
    // Fit plane for each super pixel
    fit_plane_for_cells_KernelFunc<<<block_rect, thread_rect>>>(
        points, super_pixel_id_image, base_vectors, super_pixels,
        accumulate_super_pixels, cell_hessain_upper, cell_nabla,
        super_pixel_width, number_of_block_per_line);

    // Solve x = (J'J)^-1 J'r and update super pixel params
    update_super_pixel_params_KenrelFunc<<<block_rect, 1>>>(
        accumulate_super_pixels, cell_hessain_upper, cell_nabla, base_vectors,
        super_pixels);
  } else {
    // Fit plane for each super pixel
    fit_plane_for_cells_KernelFunc_2<<<block_rect, thread_rect>>>(
        points, super_pixel_id_image, base_vectors, super_pixels,
        accumulate_super_pixels, cell_hessain_upper, cell_nabla,
        super_pixel_width, number_of_block_per_line);

    // Solve x = (J'J)^-1 J'r and update super pixel params
    update_super_pixel_params_KenrelFunc_2<<<block_rect, 1>>>(
        accumulate_super_pixels, cell_hessain_upper, cell_nabla, base_vectors,
        super_pixels);
  }
}

//
#define OUTLAYER_BAND_COEFF 1.0f
__global__ void eliminate_planar_cell_outlayers_KernelFunc(
    const My_Type::Vector3f *points, Sensor_params sensor_params,
    int *super_pixel_id_image, Super_pixel *super_pixels, int super_pixel_width,
    int number_of_block_per_line) {
  // Important! this reduce many usefuless atomic operations (speed up 1.5x)
  __shared__ bool block_need_add;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_need_add = false;
  //
  bool point_need_add = false;
  My_Type::Vector3f point_vec(0.0f);

  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  if (offset_x < super_pixel_width * 3 && offset_y < super_pixel_width * 3) {
    // compute the start of the search window
    int start_x = blockIdx.x * super_pixel_width - super_pixel_width;
    int start_y = blockIdx.y * super_pixel_width - super_pixel_width;
    //
    int u = start_x + offset_x;
    int v = start_y + offset_y;
    //
    int image_width = blockDim.x * gridDim.x;
    int image_height = blockDim.y * gridDim.y;
    int pixel_index = u + v * image_width;

    if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
      if (super_pixel_id_image[pixel_index] == super_pixel_id) {
        point_vec = points[pixel_index];
        point_need_add = true;
      }
    }
  }
  // No point need add in this block
  __syncthreads();
  if (!block_need_add) return;

  //
  My_Type::Vector3f weight_center, super_pixel_normal;
  weight_center.x = super_pixels[super_pixel_id].px;
  weight_center.y = super_pixels[super_pixel_id].py;
  weight_center.z = super_pixels[super_pixel_id].pz;
  super_pixel_normal.x = super_pixels[super_pixel_id].nx;
  super_pixel_normal.y = super_pixels[super_pixel_id].ny;
  super_pixel_normal.z = super_pixels[super_pixel_id].nz;
  My_Type::Vector3f diff_vec = point_vec - weight_center;
  float projection_diff = diff_vec.dot(super_pixel_normal);
  //
  float outlyer_band =
      OUTLAYER_BAND_COEFF * sensor_params.sensor_noise_ratio * point_vec.z;

  // TODO : ...
}

//
__global__ void generate_cells_info_KernelFunc(
    const Super_pixel *super_pixels, const Sensor_params sensor_params,
    const int super_pixel_width, Cell_info *cell_info_mat) {
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;

  //
  Super_pixel temp_sp = super_pixels[super_pixel_id];
  if (temp_sp.valid_pixel_number >= 32 && temp_sp.is_planar_cell) {
    Cell_info temp_cell;
    temp_cell.nx = temp_sp.nx;
    temp_cell.ny = temp_sp.ny;
    temp_cell.nz = temp_sp.nz;
    temp_cell.x = temp_sp.px;
    temp_cell.y = temp_sp.py;
    temp_cell.z = temp_sp.pz;
    temp_cell.counter = temp_sp.valid_pixel_number;

    // Ray direction
    My_Type::Vector3f ray_direction, normal_vec;
    normal_vec.x = temp_sp.nx;
    normal_vec.y = temp_sp.ny;
    normal_vec.z = temp_sp.nz;
    ray_direction.x = temp_sp.px;
    ray_direction.y = temp_sp.py;
    ray_direction.z = temp_sp.pz;
    ray_direction.normlize();
    float projection_factor = fabsf(normal_vec.dot(ray_direction));
    if (projection_factor > 0.1f) {
      //
      float area = (float)temp_sp.pz / sensor_params.sensor_fx;
      area *= area;
      area /= projection_factor; /* projection warp */
      area *= (float)temp_sp.valid_pixel_number;
      temp_cell.area = area;
      //
      temp_cell.is_valid_cell = true;
      //
      cell_info_mat[super_pixel_id] = temp_cell;
    } else {
      temp_cell.is_valid_cell = false;
      //
      cell_info_mat[super_pixel_id] = temp_cell;
    }
  }
}
//
void generate_cells_info_CUDA(dim3 block_rect, dim3 thread_rect,
                              const Super_pixel *super_pixels,
                              const Sensor_params sensor_params,
                              const int super_pixel_width,
                              Cell_info *cell_info_mat) {
  generate_cells_info_KernelFunc<<<block_rect, thread_rect>>>(
      super_pixels, sensor_params, super_pixel_width, cell_info_mat);
}

// Re-label super pixels
__global__ void relabel_super_pixels_KernelFunc(const Cell_info *cell_info_mat,
                                                int *super_pixel_id_image,
                                                int *plane_id_image,
                                                int super_pixel_width,
                                                int number_of_block_per_line) {
  // Relative block position of this super pixel
  int relative_block_x = blockIdx.z % number_of_block_per_line;
  int relative_block_y = blockIdx.z / number_of_block_per_line;
  //
  int offset_x = relative_block_x * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.x;
  int offset_y = relative_block_y * SUPER_PIXEL_BLOCK_WIDTH + threadIdx.y;

  //
  int super_pixel_id = blockIdx.x + blockIdx.y * gridDim.x;
  if (offset_x < super_pixel_width * 3 && offset_y < super_pixel_width * 3) {
    // compute the start of the search window
    int start_x = blockIdx.x * super_pixel_width - super_pixel_width;
    int start_y = blockIdx.y * super_pixel_width - super_pixel_width;
    //
    int u = start_x + offset_x;
    int v = start_y + offset_y;
    //
    int image_width = blockDim.x * gridDim.x;
    int image_height = blockDim.y * gridDim.y;
    int pixel_index = u + v * image_width;

    if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
      if (super_pixel_id_image[pixel_index] == super_pixel_id) {
        int plane_id = cell_info_mat[super_pixel_id].plane_index;
        plane_id_image[pixel_index] = plane_id;
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //{	printf("%d, %d\n", pixel_index, plane_id);	}
      }
    }
  }
}
//
void relabel_super_pixels_CUDA(dim3 block_rect, dim3 thread_rect,
                               const Cell_info *cell_info_mat,
                               int *super_pixel_id_image, int *plane_id_image,
                               int super_pixel_width,
                               int number_of_block_per_line) {
  relabel_super_pixels_KernelFunc<<<block_rect, thread_rect>>>(
      cell_info_mat, super_pixel_id_image, plane_id_image, super_pixel_width,
      number_of_block_per_line);
}
