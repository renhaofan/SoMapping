

//
#include "Mesh_generator_KernelFunc.cuh"

//
#include "Mesh_structure.h"
//
#include <float.h>

#include "Map_engine/Plane_pixel_device_interface.cuh"
#include "Map_engine/Voxel_device_interface.cuh"
#include "OurLib/device_color_table.cuh"

//
__global__ void find_alllocated_entries_KernelFunc(
    const HashEntry *entries, int max_number_of_entries,
    HashEntry *allocated_entries, int *number_of_allocated_entries) {
  bool is_allocated = true, is_valid_entry = true;
  int entry_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Validate memory access
  if (entry_index >= max_number_of_entries) is_valid_entry = false;

  // Load entry
  HashEntry temp_entry = entries[entry_index];
  if (temp_entry.ptr < 0) is_allocated = false;

  // Mark allocated entries
  __shared__ HashEntry entry_buffer[256];
  __shared__ int allocated_number;
  allocated_number = 0;
  __syncthreads();
  if (is_allocated && is_valid_entry) {
    int buffer_index = atomicAdd(&allocated_number, 1);
    entry_buffer[buffer_index] = temp_entry;
  }

  // Atomic get allocated_array offset
  __syncthreads();
  __shared__ int allocated_entry_offset;
  if (threadIdx.x == 0) {
    allocated_entry_offset =
        atomicAdd(number_of_allocated_entries, allocated_number);
  }

  // Save allocated entries
  __syncthreads();
  if (threadIdx.x < allocated_number) {
    int allocated_entry_index = allocated_entry_offset + threadIdx.x;
    allocated_entries[allocated_entry_index] = entry_buffer[threadIdx.x];
  }
}
//
void find_alllocated_entries_CUDA(dim3 block_rect, dim3 thread_rect,
                                  const HashEntry *entries,
                                  int max_number_of_entries,
                                  HashEntry *allocated_entries,
                                  int *number_of_allocated_entries) {
  find_alllocated_entries_KernelFunc<<<block_rect, thread_rect>>>(
      entries, max_number_of_entries, allocated_entries,
      number_of_allocated_entries);
}

//
__global__ void find_nonplanar_blocks_KernelFunc(
    const HashEntry *entries, const HashEntry *allocated_entries,
    const Voxel_f *voxel_block_array, HashEntry *nonplanar_entries,
    int *number_of_nonplanar_block) {
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int entry_index = blockIdx.x;
  HashEntry temp_entry = allocated_entries[entry_index];

  //
  My_Type::Vector3f block_offset;
  block_offset.x =
      (float)temp_entry.position[0] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
  block_offset.y =
      (float)temp_entry.position[1] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
  block_offset.z =
      (float)temp_entry.position[2] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
  //
  My_Type::Vector3f point_vec;
  point_vec.x = threadIdx.x * VOXEL_SIZE + HALF_VOXEL_SIZE;
  point_vec.y = threadIdx.y * VOXEL_SIZE + HALF_VOXEL_SIZE;
  point_vec.z = threadIdx.z * VOXEL_SIZE + HALF_VOXEL_SIZE;
  point_vec += block_offset;

  //
  bool is_valid_voxel = true;
  int voxel_index =
      get_voxel_index_neighbor(point_vec.x, point_vec.y, point_vec.z, entries);
  if (voxel_index < 0) is_valid_voxel = false;

  //
  Voxel_f temp_voxel;
  if (is_valid_voxel) {
    temp_voxel = voxel_block_array[voxel_index];
    if (temp_voxel.weight < MIN_RAYCAST_WEIGHT) is_valid_voxel = false;
  }
  if (is_valid_voxel) {
    //
    float surface_voxel_sdf = VOXEL_SIZE / TRUNCATED_BAND * 0.707f;
    if (fabsf(temp_voxel.sdf) >= surface_voxel_sdf) is_valid_voxel = false;
  }

  //
  bool is_nonplanar_voxel = true;
  if (is_valid_voxel) {
    if (temp_voxel.plane_index != 0) is_nonplanar_voxel = false;
  }

  //
  __shared__ int surface_voxel_number[512], nonplanar_voxel_number[512];
  if (is_valid_voxel) {
    surface_voxel_number[tid] = 1;
    if (is_nonplanar_voxel) {
      nonplanar_voxel_number[tid] = 1;
    } else {
      nonplanar_voxel_number[tid] = 0;
    }
  } else {
    surface_voxel_number[tid] = 0;
    nonplanar_voxel_number[tid] = 0;
  }

  //
  block_512_reduce(surface_voxel_number, tid);
  block_512_reduce(nonplanar_voxel_number, tid);

  //
  if (tid == 0) {
    if (nonplanar_voxel_number[0] == 0) return;
    if (surface_voxel_number[0] == 0) return;

    //
    float nonplanar_ratio =
        (float)nonplanar_voxel_number[0] / (float)surface_voxel_number[0];
    const float nonplanar_threshold = 0.2f;
    //
    if (nonplanar_ratio > nonplanar_threshold) {
      int nonplanar_entry_index = atomicAdd(number_of_nonplanar_block, 1);
      nonplanar_entries[nonplanar_entry_index] = temp_entry;
    }
  }
}
//
void find_nonplanar_blocks_CUDA(dim3 block_rect, dim3 thread_rect,
                                const HashEntry *entries,
                                const HashEntry *allocated_entries,
                                const Voxel_f *voxel_block_array,
                                HashEntry *nonplanar_entries,
                                int *number_of_nonplanar_block) {
  find_nonplanar_blocks_KernelFunc<<<block_rect, thread_rect>>>(
      entries, allocated_entries, voxel_block_array, nonplanar_entries,
      number_of_nonplanar_block);
}

// Interpolate SDF value
__device__ inline bool get_sdf_from_voxel_block(
    const HashEntry &this_entry, const HashEntry *entries,
    const Voxel_f *voxel_block_array, My_Type::Vector3f *sdf_position,
    float *sdf_value) {
  My_Type::Vector3f block_offset;
  block_offset.x =
      (float)this_entry.position[0] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
  block_offset.y =
      (float)this_entry.position[1] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
  block_offset.z =
      (float)this_entry.position[2] * VOXEL_BLOCK_WDITH * VOXEL_SIZE;

  //
  bool is_valid_cube;
  // 0 0 0
  sdf_position[0].x = block_offset.x + (threadIdx.x + 0) * VOXEL_SIZE;
  sdf_position[0].y = block_offset.y + (threadIdx.y + 0) * VOXEL_SIZE;
  sdf_position[0].z = block_offset.z + (threadIdx.z + 0) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[0].x, sdf_position[0].y,
                                       sdf_position[0].z, entries,
                                       voxel_block_array, sdf_value[0]);
  if (!is_valid_cube) return false;
  // 0 0 1
  sdf_position[1].x = block_offset.x + (threadIdx.x + 1) * VOXEL_SIZE;
  sdf_position[1].y = block_offset.y + (threadIdx.y + 0) * VOXEL_SIZE;
  sdf_position[1].z = block_offset.z + (threadIdx.z + 0) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[1].x, sdf_position[1].y,
                                       sdf_position[1].z, entries,
                                       voxel_block_array, sdf_value[1]);
  if (!is_valid_cube) return false;
  // 0 1 1
  sdf_position[2].x = block_offset.x + (threadIdx.x + 1) * VOXEL_SIZE;
  sdf_position[2].y = block_offset.y + (threadIdx.y + 1) * VOXEL_SIZE;
  sdf_position[2].z = block_offset.z + (threadIdx.z + 0) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[2].x, sdf_position[2].y,
                                       sdf_position[2].z, entries,
                                       voxel_block_array, sdf_value[2]);
  if (!is_valid_cube) return false;
  // 0 1 0
  sdf_position[3].x = block_offset.x + (threadIdx.x + 0) * VOXEL_SIZE;
  sdf_position[3].y = block_offset.y + (threadIdx.y + 1) * VOXEL_SIZE;
  sdf_position[3].z = block_offset.z + (threadIdx.z + 0) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[3].x, sdf_position[3].y,
                                       sdf_position[3].z, entries,
                                       voxel_block_array, sdf_value[3]);
  if (!is_valid_cube) return false;
  // 1 0 0
  sdf_position[4].x = block_offset.x + (threadIdx.x + 0) * VOXEL_SIZE;
  sdf_position[4].y = block_offset.y + (threadIdx.y + 0) * VOXEL_SIZE;
  sdf_position[4].z = block_offset.z + (threadIdx.z + 1) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[4].x, sdf_position[4].y,
                                       sdf_position[4].z, entries,
                                       voxel_block_array, sdf_value[4]);
  if (!is_valid_cube) return false;
  // 1 0 1
  sdf_position[5].x = block_offset.x + (threadIdx.x + 1) * VOXEL_SIZE;
  sdf_position[5].y = block_offset.y + (threadIdx.y + 0) * VOXEL_SIZE;
  sdf_position[5].z = block_offset.z + (threadIdx.z + 1) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[5].x, sdf_position[5].y,
                                       sdf_position[5].z, entries,
                                       voxel_block_array, sdf_value[5]);
  if (!is_valid_cube) return false;
  // 1 1 1
  sdf_position[6].x = block_offset.x + (threadIdx.x + 1) * VOXEL_SIZE;
  sdf_position[6].y = block_offset.y + (threadIdx.y + 1) * VOXEL_SIZE;
  sdf_position[6].z = block_offset.z + (threadIdx.z + 1) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[6].x, sdf_position[6].y,
                                       sdf_position[6].z, entries,
                                       voxel_block_array, sdf_value[6]);
  if (!is_valid_cube) return false;
  // 1 1 0
  sdf_position[7].x = block_offset.x + (threadIdx.x + 0) * VOXEL_SIZE;
  sdf_position[7].y = block_offset.y + (threadIdx.y + 1) * VOXEL_SIZE;
  sdf_position[7].z = block_offset.z + (threadIdx.z + 1) * VOXEL_SIZE;
  is_valid_cube = get_sdf_interpolated(sdf_position[7].x, sdf_position[7].y,
                                       sdf_position[7].z, entries,
                                       voxel_block_array, sdf_value[7]);
  if (!is_valid_cube) return false;

  return true;
}

//
__device__ inline My_Type::Vector3f bilinear_interpolate_by_sdf(
    const My_Type::Vector3f &vertex_1, const My_Type::Vector3f &vertex_2,
    float sdf_1, float sdf_2) {
  if (fabs(0.0f - sdf_1) < 0.00001f) return vertex_1;
  if (fabs(0.0f - sdf_2) < 0.00001f) return vertex_2;
  if (fabs(sdf_1 - sdf_2) < 0.00001f) return vertex_1;

  return vertex_1 + ((0.0f - sdf_1) / (sdf_2 - sdf_1)) * (vertex_2 - vertex_1);
}

//
__global__ void generate_triangle_mesh_KernelFunc(
    const HashEntry *entries, const HashEntry *allocated_entries,
    const Voxel_f *voxel_block_array, My_Type::Vector3f *triangles,
    int *number_of_triangles, int max_number_of_triangles) {
  //
  HashEntry temp_entry = allocated_entries[blockIdx.x];

  // Get interpolated SDF value of 8 vertex in the voxel cube
  float sdf_value[8];
  My_Type::Vector3f sdf_position[8];
  bool is_valid_cube = get_sdf_from_voxel_block(
      temp_entry, entries, voxel_block_array, sdf_position, sdf_value);

  //
  int cube_index = 0;
  if (is_valid_cube) {
    if (sdf_value[0] < 0) cube_index |= 1;
    if (sdf_value[1] < 0) cube_index |= 2;
    if (sdf_value[2] < 0) cube_index |= 4;
    if (sdf_value[3] < 0) cube_index |= 8;
    if (sdf_value[4] < 0) cube_index |= 16;
    if (sdf_value[5] < 0) cube_index |= 32;
    if (sdf_value[6] < 0) cube_index |= 64;
    if (sdf_value[7] < 0) cube_index |= 128;
    if (edgeTable[cube_index] == 0) is_valid_cube = false;
  }

  //
  My_Type::Vector3f vertex_list[12];
  if (is_valid_cube) {
    if (edgeTable[cube_index] & 1)
      vertex_list[0] = bilinear_interpolate_by_sdf(
          sdf_position[0], sdf_position[1], sdf_value[0], sdf_value[1]);
    if (edgeTable[cube_index] & 2)
      vertex_list[1] = bilinear_interpolate_by_sdf(
          sdf_position[1], sdf_position[2], sdf_value[1], sdf_value[2]);
    if (edgeTable[cube_index] & 4)
      vertex_list[2] = bilinear_interpolate_by_sdf(
          sdf_position[2], sdf_position[3], sdf_value[2], sdf_value[3]);
    if (edgeTable[cube_index] & 8)
      vertex_list[3] = bilinear_interpolate_by_sdf(
          sdf_position[3], sdf_position[0], sdf_value[3], sdf_value[0]);
    if (edgeTable[cube_index] & 16)
      vertex_list[4] = bilinear_interpolate_by_sdf(
          sdf_position[4], sdf_position[5], sdf_value[4], sdf_value[5]);
    if (edgeTable[cube_index] & 32)
      vertex_list[5] = bilinear_interpolate_by_sdf(
          sdf_position[5], sdf_position[6], sdf_value[5], sdf_value[6]);
    if (edgeTable[cube_index] & 64)
      vertex_list[6] = bilinear_interpolate_by_sdf(
          sdf_position[6], sdf_position[7], sdf_value[6], sdf_value[7]);
    if (edgeTable[cube_index] & 128)
      vertex_list[7] = bilinear_interpolate_by_sdf(
          sdf_position[7], sdf_position[4], sdf_value[7], sdf_value[4]);
    if (edgeTable[cube_index] & 256)
      vertex_list[8] = bilinear_interpolate_by_sdf(
          sdf_position[0], sdf_position[4], sdf_value[0], sdf_value[4]);
    if (edgeTable[cube_index] & 512)
      vertex_list[9] = bilinear_interpolate_by_sdf(
          sdf_position[1], sdf_position[5], sdf_value[1], sdf_value[5]);
    if (edgeTable[cube_index] & 1024)
      vertex_list[10] = bilinear_interpolate_by_sdf(
          sdf_position[2], sdf_position[6], sdf_value[2], sdf_value[6]);
    if (edgeTable[cube_index] & 2048)
      vertex_list[11] = bilinear_interpolate_by_sdf(
          sdf_position[3], sdf_position[7], sdf_value[3], sdf_value[7]);
  }

  //
  if (is_valid_cube) {
    for (int i = 0; triangleTable[cube_index][i] != -1; i += 3) {
      int triangleId = atomicAdd(number_of_triangles, 1);

      if (triangleId < max_number_of_triangles - 1) {
        triangles[triangleId * 3 + 0] =
            vertex_list[triangleTable[cube_index][i + 0]];
        triangles[triangleId * 3 + 1] =
            vertex_list[triangleTable[cube_index][i + 1]];
        triangles[triangleId * 3 + 2] =
            vertex_list[triangleTable[cube_index][i + 2]];
      }
    }
  }
}
//
void generate_triangle_mesh_CUDA(dim3 block_rect, dim3 thread_rect,
                                 const HashEntry *entries,
                                 const HashEntry *allocated_entries,
                                 const Voxel_f *voxel_block_array,
                                 My_Type::Vector3f *triangles,
                                 int *number_of_triangles,
                                 int max_number_of_triangles) {
  generate_triangle_mesh_KernelFunc<<<block_rect, thread_rect>>>(
      entries, allocated_entries, voxel_block_array, triangles,
      number_of_triangles, max_number_of_triangles);
}

//
__global__ void generate_vertex_normals_KernelFunc(
    const My_Type::Vector3f *vertex_array, const HashEntry *entries,
    const Voxel_f *voxel_block_array, My_Type::Vector3f *vertex_normals) {
  int vertex_index = threadIdx.x + blockDim.x * blockIdx.x;

  //
  My_Type::Vector3f temp_vertex = vertex_array[vertex_index];
  My_Type::Vector3f normal_vector;

  // compute normal vector
  bool is_valid_normal =
      interpolate_normal_by_sdf(temp_vertex.x, temp_vertex.y, temp_vertex.z,
                                entries, voxel_block_array, normal_vector);

  if (!is_valid_normal) normal_vector = 0;
  //
  vertex_normals[vertex_index] = normal_vector;
}
//
void generate_vertex_normals_CUDA(dim3 block_rect, dim3 thread_rect,
                                  const My_Type::Vector3f *vertex_array,
                                  const HashEntry *entries,
                                  const Voxel_f *voxel_block_array,
                                  My_Type::Vector3f *vertex_normals) {
  generate_vertex_normals_KernelFunc<<<block_rect, thread_rect>>>(
      vertex_array, entries, voxel_block_array, vertex_normals);
}

//
__global__ void generate_vertex_color_KernelFunc(
    const My_Type::Vector3f *vertex_array, int number_of_vertex,
    const HashEntry *entries, const Voxel_f *voxel_block_array,
    My_Type::Vector4uc *vertex_color_array) {
  int vertex_index = threadIdx.x + blockDim.x * blockIdx.x;
  if (number_of_vertex <= vertex_index) return;

  //
  My_Type::Vector3f temp_vertex = vertex_array[vertex_index];
  My_Type::Vector4uc vertex_color;

  // compute normal vector
  int voxel_index = get_voxel_index_neighbor(temp_vertex.x, temp_vertex.y,
                                             temp_vertex.z, entries);
  //
  bool is_planar_voxel = false;
  int plane_label = 0;
  if (voxel_index >= 0) {
    plane_label = voxel_block_array[voxel_index].plane_index;
    if (plane_label > 0) is_planar_voxel = true;
  }

  if (is_planar_voxel) {
    int plane_color_index = (plane_label * PLANE_COLOR_STEP) % COLOR_NUM;

    // Alpha
    vertex_color.a = 0xFF;
    // B G R
    vertex_color.b = (unsigned char)color_table_Blue[plane_color_index];
    vertex_color.g = (unsigned char)color_table_Green[plane_color_index];
    vertex_color.r = (unsigned char)color_table_Red[plane_color_index];
  } else {
    vertex_color = 0xFF;
  }
  //
  vertex_color_array[vertex_index] = vertex_color;
}
//
void generate_vertex_color_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *vertex_array,
                                int number_of_vertex, const HashEntry *entries,
                                const Voxel_f *voxel_block_array,
                                My_Type::Vector4uc *vertex_color_array) {
  generate_vertex_color_KernelFunc<<<block_rect, thread_rect>>>(
      vertex_array, number_of_vertex, entries, voxel_block_array,
      vertex_color_array);
}
//
__global__ void generate_vertex_color_KernelFunc(
    const My_Type::Vector3f *vertex_array, int number_of_vertex,
    const HashEntry *entries, const Voxel_f *voxel_block_array,
    const My_Type::Vector2i *plane_label_mapper,
    My_Type::Vector4uc *vertex_color_array) {
  int vertex_index = threadIdx.x + blockDim.x * blockIdx.x;
  if (number_of_vertex <= vertex_index) return;

  //
  My_Type::Vector3f temp_vertex = vertex_array[vertex_index];
  My_Type::Vector4uc vertex_color;

  // compute normal vector
  int voxel_index = get_voxel_index_neighbor(temp_vertex.x, temp_vertex.y,
                                             temp_vertex.z, entries);
  //
  bool is_planar_voxel = false;
  int plane_label = 0;
  if (voxel_index >= 0) {
    plane_label = voxel_block_array[voxel_index].plane_index;
    if (plane_label > 0) is_planar_voxel = true;
  }

  if (is_planar_voxel) {
    if (plane_label < 0 || plane_label > 127)
      printf("plane_label = %d\n", plane_label);
    if (plane_label_mapper[plane_label].y < 0 ||
        plane_label_mapper[plane_label].y > 127)
      printf("plane_label_mapper[plane_label].y = %d\n",
             plane_label_mapper[plane_label].y);
    if (plane_label_mapper[plane_label].y > 0)
      plane_label = plane_label_mapper[plane_label].y;

    int plane_color_index = (plane_label * PLANE_COLOR_STEP) % COLOR_NUM;

    // Alpha
    vertex_color.a = 0xFF;
    // B G R
    vertex_color.b = (unsigned char)color_table_Blue[plane_color_index];
    vertex_color.g = (unsigned char)color_table_Green[plane_color_index];
    vertex_color.r = (unsigned char)color_table_Red[plane_color_index];
  } else {
    vertex_color = 0xFF;
  }
  //
  vertex_color_array[vertex_index] = vertex_color;
}
//
void generate_vertex_color_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *vertex_array,
                                int number_of_vertex, const HashEntry *entries,
                                const Voxel_f *voxel_block_array,
                                const My_Type::Vector2i *plane_label_mapper,
                                My_Type::Vector4uc *vertex_color_array) {
  generate_vertex_color_KernelFunc<<<block_rect, thread_rect>>>(
      vertex_array, number_of_vertex, entries, voxel_block_array,
      plane_label_mapper, vertex_color_array);
}

//
__device__ void compute_vertex_of_plane_pixel(int u_offset, int v_offset,
                                              float plane_distance, float diff,
                                              Plane_coordinate plane_coordinate,
                                              My_Type::Vector3f &pixel_point) {
  //
  float px = u_offset * PLANE_PIXEL_SIZE;
  float py = v_offset * PLANE_PIXEL_SIZE;
  float distance = -plane_distance;
  //
  pixel_point.x = px * plane_coordinate.x_vec.x +
                  py * plane_coordinate.y_vec.x +
                  distance * plane_coordinate.z_vec.x;
  pixel_point.y = px * plane_coordinate.x_vec.y +
                  py * plane_coordinate.y_vec.y +
                  distance * plane_coordinate.z_vec.y;
  pixel_point.z = px * plane_coordinate.x_vec.z +
                  py * plane_coordinate.y_vec.z +
                  distance * plane_coordinate.z_vec.z;
  pixel_point += diff * plane_coordinate.z_vec;
}
//
__global__ void generate_triangle_mesh_from_plane_KernelFunc(
    Plane_info model_plane, Plane_coordinate plane_coordinate,
    const PlaneHashEntry *plane_entries, const Plane_pixel *plane_pixel_array,
    My_Type::Vector3f *vertex_array, int *triangle_counter) {
  //
  bool is_valid_pixel = true;
  //
  int entry_index = blockIdx.x;
  PlaneHashEntry temp_entry = plane_entries[entry_index];
  if (temp_entry.ptr < 0) is_valid_pixel = false;

  //
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int pixel_index_00, pixel_index_01, pixel_index_10, pixel_index_11, u_offset,
      v_offset;
  if (is_valid_pixel) {
    u_offset = temp_entry.position[0] * PLANE_PIXEL_BLOCK_WIDTH + threadIdx.x;
    v_offset = temp_entry.position[1] * PLANE_PIXEL_BLOCK_WIDTH + threadIdx.y;

    //
    pixel_index_00 = get_pixel_index(u_offset, v_offset, plane_entries);
    if (pixel_index_00 < 0) is_valid_pixel = false;
    pixel_index_01 = get_pixel_index(u_offset + 1, v_offset, plane_entries);
    if (pixel_index_01 < 0) is_valid_pixel = false;
    pixel_index_10 = get_pixel_index(u_offset, v_offset + 1, plane_entries);
    if (pixel_index_10 < 0) is_valid_pixel = false;
    pixel_index_11 = get_pixel_index(u_offset + 1, v_offset + 1, plane_entries);
    if (pixel_index_11 < 0) is_valid_pixel = false;
    // if (tid == 0)printf("%d, %d = %d\n", u_offset, v_offset, pixel_index_00);
  }

  __shared__ int number_of_cache_triangles;
  __shared__ My_Type::Vector3f vertex_cache[PLANE_PIXEL_BLOCK_SIZE * 6];
  number_of_cache_triangles = 0;
  __syncthreads();  //__threadfence_block()
  //
  float diff_00, diff_01, diff_10, diff_11;
  if (is_valid_pixel) {
    //
    diff_00 = plane_pixel_array[pixel_index_00].diff;
    diff_01 = plane_pixel_array[pixel_index_01].diff;
    diff_10 = plane_pixel_array[pixel_index_10].diff;
    diff_11 = plane_pixel_array[pixel_index_11].diff;
    if (diff_00 == FLT_MAX || diff_01 == FLT_MAX || diff_10 == FLT_MAX ||
        diff_11 == FLT_MAX)
      is_valid_pixel = false;
  }

  //
  if (is_valid_pixel) {
    //
    My_Type::Vector3f point_00, point_01, point_10, point_11;
    compute_vertex_of_plane_pixel(u_offset, v_offset, model_plane.d, diff_00,
                                  plane_coordinate, point_00);
    compute_vertex_of_plane_pixel(u_offset + 1, v_offset, model_plane.d,
                                  diff_01, plane_coordinate, point_01);
    compute_vertex_of_plane_pixel(u_offset, v_offset + 1, model_plane.d,
                                  diff_10, plane_coordinate, point_10);
    compute_vertex_of_plane_pixel(u_offset + 1, v_offset + 1, model_plane.d,
                                  diff_11, plane_coordinate, point_11);

    //
    int cache_triangle_index = atomicAdd(&number_of_cache_triangles, 2);
    // T1
    vertex_cache[cache_triangle_index * 3 + 0] = point_00;
    vertex_cache[cache_triangle_index * 3 + 1] = point_10;
    vertex_cache[cache_triangle_index * 3 + 2] = point_01;
    // T2
    vertex_cache[cache_triangle_index * 3 + 3] = point_11;
    vertex_cache[cache_triangle_index * 3 + 4] = point_01;
    vertex_cache[cache_triangle_index * 3 + 5] = point_10;
  }
  __syncthreads();

  //
  __shared__ int vertex_index_offset_s;
  if (tid == 0) {
    // if (tid == 0 && number_of_cache_triangles> 0)printf("triangle_counter =
    // %d\n", number_of_cache_triangles);
    vertex_index_offset_s =
        atomicAdd(triangle_counter, number_of_cache_triangles);
  }
  __syncthreads();

  //
  int copy_thread_number = number_of_cache_triangles / 2;
  if (tid < copy_thread_number) {
    int vertex_index_offset = vertex_index_offset_s * 3;
    //
    vertex_array[vertex_index_offset + tid + 0 * copy_thread_number] =
        vertex_cache[tid + 0 * copy_thread_number];
    __threadfence();
    vertex_array[vertex_index_offset + tid + 1 * copy_thread_number] =
        vertex_cache[tid + 1 * copy_thread_number];
    __threadfence();
    vertex_array[vertex_index_offset + tid + 2 * copy_thread_number] =
        vertex_cache[tid + 2 * copy_thread_number];
    __threadfence();
    vertex_array[vertex_index_offset + tid + 3 * copy_thread_number] =
        vertex_cache[tid + 3 * copy_thread_number];
    __threadfence();
    vertex_array[vertex_index_offset + tid + 4 * copy_thread_number] =
        vertex_cache[tid + 4 * copy_thread_number];
    __threadfence();
    vertex_array[vertex_index_offset + tid + 5 * copy_thread_number] =
        vertex_cache[tid + 5 * copy_thread_number];
    __threadfence();
  }
}
//
void generate_triangle_mesh_from_plane_CUDA(
    dim3 block_rect, dim3 thread_rect, Plane_info model_plane,
    Plane_coordinate plane_coordinate, const PlaneHashEntry *plane_entries,
    const Plane_pixel *plane_pixel_array, My_Type::Vector3f *vertex_array,
    int *triangle_counter) {
  generate_triangle_mesh_from_plane_KernelFunc<<<block_rect, thread_rect>>>(
      model_plane, plane_coordinate, plane_entries, plane_pixel_array,
      vertex_array, triangle_counter);
}

//
__global__ void copy_mesh_to_global_map_KernelFunc(
    const My_Type::Vector3f *src_vertex_array,
    const My_Type::Vector3f *src_normal_array,
    const My_Type::Matrix44f submap_pose, int number_of_vertex,
    My_Type::Vector3f *dst_vertex_array, My_Type::Vector3f *dst_normal_array) {
  int vertex_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (vertex_id >= number_of_vertex) return;

  // Load src_data
  My_Type::Vector3f src_vertex = src_vertex_array[vertex_id];
  My_Type::Vector3f src_normal = src_normal_array[vertex_id];

  // Transfer to dst coordinate
  My_Type::Vector3f dst_vertex, dst_normal;
  dst_vertex.x = submap_pose[0] * src_vertex.x + submap_pose[4] * src_vertex.y +
                 submap_pose[8] * src_vertex.z + submap_pose[12];
  dst_vertex.y = submap_pose[1] * src_vertex.x + submap_pose[5] * src_vertex.y +
                 submap_pose[9] * src_vertex.z + submap_pose[13];
  dst_vertex.z = submap_pose[2] * src_vertex.x + submap_pose[6] * src_vertex.y +
                 submap_pose[10] * src_vertex.z + submap_pose[14];
  dst_normal.x = submap_pose[0] * src_normal.x + submap_pose[4] * src_normal.y +
                 submap_pose[8] * src_normal.z;
  dst_normal.y = submap_pose[1] * src_normal.x + submap_pose[5] * src_normal.y +
                 submap_pose[9] * src_normal.z;
  dst_normal.z = submap_pose[2] * src_normal.x + submap_pose[6] * src_normal.y +
                 submap_pose[10] * src_normal.z;

  // Save dst data
  dst_vertex_array[vertex_id] = dst_vertex;
  dst_normal_array[vertex_id] = dst_normal;
}
//
void copy_mesh_to_global_map_CUDA(dim3 block_rect, dim3 thread_rect,
                                  const My_Type::Vector3f *src_vertex_array,
                                  const My_Type::Vector3f *src_normal_array,
                                  const My_Type::Matrix44f submap_pose,
                                  int number_of_vertex,
                                  My_Type::Vector3f *dst_vertex_array,
                                  My_Type::Vector3f *dst_normal_array) {
  copy_mesh_to_global_map_KernelFunc<<<block_rect, thread_rect>>>(
      src_vertex_array, src_normal_array, submap_pose, number_of_vertex,
      dst_vertex_array, dst_normal_array);
}
