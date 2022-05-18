

#include "Render_engine.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// C/C++ IO
#include <iostream>
#include <stdio.h>
using namespace std;

//
#include "SLAM_system/SLAM_system_settings.h"
#include "UI_engine/UI_parameters.h"

//
#include "Render_KernelFunc.cuh"

#pragma region(CUDA memory operation for hierarchy image)
// CUDA memory allocation for hierarchy image
template <typename T>
void allocate_CUDA_memory_for_hierarchy(Hierarchy_image<T> &hierarcgy_image) {
  for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers;
       layer_id++) {
    checkCudaErrors(cudaMalloc((void **)&(hierarcgy_image.data_ptrs[layer_id]),
                               hierarcgy_image.size[layer_id].width *
                                   hierarcgy_image.size[layer_id].height *
                                   sizeof(T)));
    checkCudaErrors(cudaMemset(hierarcgy_image.data_ptrs[layer_id], 0x00,
                               hierarcgy_image.size[layer_id].width *
                                   hierarcgy_image.size[layer_id].height *
                                   sizeof(T)));
  }
}
// CUDA memory free for hierarchy image
template <typename T>
void release_CUDA_memory_for_hierarchy(Hierarchy_image<T> &hierarcgy_image) {
  for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers;
       layer_id++)
    checkCudaErrors(cudaFree(hierarcgy_image.data_ptrs[layer_id]));
}

#pragma endregion

#pragma region(HOST memory operation for hierarchy image)
// Host memory allocation for hierarchy image
template <typename T>
void allocate_host_memory_for_hierarchy(Hierarchy_image<T> &hierarcgy_image) {
  for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers;
       layer_id++) {
    hierarcgy_image.data_ptrs[layer_id] =
        (T *)malloc(hierarcgy_image.size[layer_id].width *
                    hierarcgy_image.size[layer_id].height * sizeof(T));
    memset(hierarcgy_image.data_ptrs[layer_id], 0x00,
           hierarcgy_image.size[layer_id].width *
               hierarcgy_image.size[layer_id].height * sizeof(T));
  }
}
// Host memory release for hierarchy image
template <typename T>
void release_host_memory_for_hierarchy(Hierarchy_image<T> &hierarcgy_image) {
  for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers;
       layer_id++)
    free(hierarcgy_image.data_ptrs[layer_id]);
}
#pragma endregion

//
Render_engine::Render_engine() {}
Render_engine::~Render_engine() {
  //------ Viewport1
  //
  checkCudaErrors(cudaFree(this->dev_scene_points));
  checkCudaErrors(cudaFree(this->dev_scene_normals));
  checkCudaErrors(cudaFree(this->dev_scene_points_weight));
  checkCudaErrors(cudaFree(this->dev_scene_plane_label));
  checkCudaErrors(cudaFree(this->dev_scene_points_color));

  //
  free(this->range_map);
  free(this->current_points);
  free(this->model_points);
  free(this->scene_points);
  free(this->scene_normals);
  free(this->scene_points_color);

  //
  release_CUDA_memory_for_hierarchy(this->dev_current_hierarchy_normal_to_draw);
  release_CUDA_memory_for_hierarchy(this->dev_model_hierarchy_normal_to_draw);
  release_host_memory_for_hierarchy(this->current_hierarchy_normal_to_draw);
  release_host_memory_for_hierarchy(this->model_hierarchy_normal_to_draw);

  //------ Viewport2
  //
  checkCudaErrors(cudaFree(this->dev_min_depth));
  checkCudaErrors(cudaFree(this->dev_max_depth));
  checkCudaErrors(cudaFree(this->dev_viewport_2_color));

  //
  free(this->viewport_2_color);

  // Pseudo plane color
  checkCudaErrors(cudaFree(this->dev_pseudo_plane_color));
  free(this->pseudo_plane_color);

  //
  free(this->enties_buffer);
  free(this->voxel_block_lines);
}

//
void Render_engine::init(My_Type::Vector2i depth_size,
                         My_Type::Vector2i scene_depth_size) {
  //
  this->depth_size = depth_size;
  this->scene_depth_size = scene_depth_size;
  // Compute raytcast range map size ( '1 + ...' for boundary)
  this->range_map_size.width =
      1 +
      (int)ceilf(
          (float)depth_size.width /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);
  this->range_map_size.height =
      1 +
      (int)ceilf(
          (float)depth_size.height /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);

  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_points),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_normals),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_points_weight),
                             this->scene_depth_size.width *
                                 this->scene_depth_size.height * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_plane_label),
                             this->scene_depth_size.width *
                                 this->scene_depth_size.height * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_points_color),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector4uc)));

  //
  this->range_map = (My_Type::Vector2f *)malloc(this->range_map_size.width *
                                                this->range_map_size.height *
                                                sizeof(My_Type::Vector2f));
  //
  this->current_points = (My_Type::Vector3f *)malloc(this->depth_size.width *
                                                     this->depth_size.height *
                                                     sizeof(My_Type::Vector3f));
  this->model_points = (My_Type::Vector3f *)malloc(this->depth_size.width *
                                                   this->depth_size.height *
                                                   sizeof(My_Type::Vector3f));
  //
  this->scene_points = (My_Type::Vector3f *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector3f));
  this->scene_normals = (My_Type::Vector3f *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector3f));
  this->scene_points_color = (My_Type::Vector4uc *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector4uc));

  //
  this->current_hierarchy_normal_to_draw.init_parameters(
      this->depth_size,
      SLAM_system_settings::instance()->image_alginment_patch_width, 3);
  this->dev_current_hierarchy_normal_to_draw.init_parameters(
      this->depth_size,
      SLAM_system_settings::instance()->image_alginment_patch_width, 3);
  this->model_hierarchy_normal_to_draw.init_parameters(
      this->depth_size,
      SLAM_system_settings::instance()->image_alginment_patch_width, 3);
  this->dev_model_hierarchy_normal_to_draw.init_parameters(
      this->depth_size,
      SLAM_system_settings::instance()->image_alginment_patch_width, 3);

  //
  allocate_CUDA_memory_for_hierarchy(
      this->dev_current_hierarchy_normal_to_draw);
  allocate_CUDA_memory_for_hierarchy(this->dev_model_hierarchy_normal_to_draw);
  //
  allocate_host_memory_for_hierarchy(this->current_hierarchy_normal_to_draw);
  allocate_host_memory_for_hierarchy(this->model_hierarchy_normal_to_draw);

  //------ Viewport2
  checkCudaErrors(cudaMalloc((void **)&(this->dev_min_depth), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_max_depth), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_viewport_2_color),
                             this->depth_size.width * this->depth_size.height *
                                 sizeof(My_Type::Vector4uc)));

  //
  checkCudaErrors(cudaMemset(this->dev_min_depth, 0x7F, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_max_depth, 0x00, sizeof(int)));

  //
  this->viewport_2_color = (My_Type::Vector4uc *)malloc(
      this->depth_size.width * this->depth_size.height *
      sizeof(My_Type::Vector4uc));

  // Pseudo plane color
  checkCudaErrors(cudaMalloc((void **)&(this->dev_pseudo_plane_color),
                             this->depth_size.width * this->depth_size.height *
                                 sizeof(My_Type::Vector4uc)));
  this->pseudo_plane_color = (My_Type::Vector4uc *)malloc(
      this->depth_size.width * this->depth_size.height *
      sizeof(My_Type::Vector4uc));

  //
  this->enties_buffer = (HashEntry *)malloc(
      (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(HashEntry));
}

//
void Render_engine::scene_viewport_reshape(My_Type::Vector2i scene_depth_size) {
  this->scene_depth_size = scene_depth_size;

  // Release CUDA memory
  checkCudaErrors(cudaFree(this->dev_scene_points));
  checkCudaErrors(cudaFree(this->dev_scene_normals));
  checkCudaErrors(cudaFree(this->dev_scene_points_weight));
  checkCudaErrors(cudaFree(this->dev_scene_plane_label));
  checkCudaErrors(cudaFree(this->dev_scene_points_color));
  // Release HOST memory
  free(this->scene_points);
  free(this->scene_normals);
  free(this->scene_points_color);

  // Re-allocation CUDA memory
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_points),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_normals),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_points_weight),
                             this->scene_depth_size.width *
                                 this->scene_depth_size.height * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_plane_label),
                             this->scene_depth_size.width *
                                 this->scene_depth_size.height * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_points_color),
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector4uc)));
  // Re-allocation HOST memory
  this->scene_points = (My_Type::Vector3f *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector3f));
  this->scene_normals = (My_Type::Vector3f *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector3f));
  this->scene_points_color = (My_Type::Vector4uc *)malloc(
      this->scene_depth_size.width * this->scene_depth_size.height *
      sizeof(My_Type::Vector4uc));
}

//
void Render_engine::render_scene_points(MainViewportRenderMode render_mode) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(
      cudaMemset(this->dev_scene_points_color, 0x00,
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector4uc)));
  //
  switch (render_mode) {
  case PHONG_RENDER: {
    //
    checkCudaErrors(cudaMemcpy(this->dev_scene_normals, this->scene_normals,
                               this->scene_depth_size.width *
                                   this->scene_depth_size.height *
                                   sizeof(My_Type::Vector3f),
                               cudaMemcpyHostToDevice));

    //
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->scene_depth_size.width / (float)thread_rect.x;
    block_rect.y = this->scene_depth_size.height / (float)thread_rect.y;
    block_rect.z = 1;
    //
    render_gypsum_CUDA(block_rect, thread_rect, this->dev_scene_normals,
                       this->dev_scene_points_color);
    // CUDA_CKECK_KERNEL;

    break;
  }
  case SDF_WEIGHT_RENDER: {
    break;
  }
  case SEMANTIC_PLANE_RENDER: {
    break;
  }
  default:
    break;
  }

  // Copy color buffer
  checkCudaErrors(
      cudaMemcpy(this->scene_points_color, this->dev_scene_points_color,
                 this->scene_depth_size.width * this->scene_depth_size.height *
                     sizeof(My_Type::Vector4uc),
                 cudaMemcpyDeviceToHost));
}

//
void Render_engine::pseudo_render_depth(
    My_Type::Vector3f *dev_raw_aligned_points) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  checkCudaErrors(cudaMemset(this->dev_min_depth, 0x7F, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_max_depth, 0x00, sizeof(int)));

  // Reduce min/max value
  {
    //
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->scene_depth_size.width / thread_rect.x;
    block_rect.y = this->scene_depth_size.height / thread_rect.y;
    block_rect.z = 1;
    //
    reduce_range_CUDA(block_rect, thread_rect, dev_raw_aligned_points,
                      this->dev_min_depth, this->dev_max_depth);
    // CUDA_CKECK_KERNEL;
  }

  // Pseudo render depth image
  {
    //
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->scene_depth_size.width / thread_rect.x;
    block_rect.y = this->scene_depth_size.height / thread_rect.y;
    block_rect.z = 1;
    pseudo_render_depth_CUDA(block_rect, thread_rect, dev_raw_aligned_points,
                             this->dev_min_depth, this->dev_max_depth,
                             this->dev_viewport_2_color);
    // CUDA_CKECK_KERNEL;
  }

  // checkCudaErrors(cudaMemcpy(&(this->min_depth), this->dev_min_depth,
  // sizeof(int), cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(&(this->max_depth), this->dev_max_depth,
  // sizeof(int), cudaMemcpyDeviceToHost)); cout << this->min_depth << "," <<
  // this->max_depth << endl;

  // Copy color buffer
  checkCudaErrors(cudaMemcpy(this->viewport_2_color, this->dev_viewport_2_color,
                             this->depth_size.width * this->depth_size.height *
                                 sizeof(My_Type::Vector4uc),
                             cudaMemcpyDeviceToHost));
}

void Render_engine::pseudo_render_plane_labels(int *dev_plane_labels) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Render plane labels
  {
    //
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->depth_size.width / (float)thread_rect.x;
    block_rect.y = this->depth_size.height / (float)thread_rect.y;
    block_rect.z = 1;
    //
    render_plane_label_CUDA(block_rect, thread_rect, dev_plane_labels,
                            this->dev_pseudo_plane_color);
    // CUDA_CKECK_KERNEL;

    // Copy out color buffer
    checkCudaErrors(
        cudaMemcpy(this->pseudo_plane_color, this->dev_pseudo_plane_color,
                   this->depth_size.width * this->depth_size.height *
                       sizeof(My_Type::Vector4uc),
                   cudaMemcpyDeviceToHost));
  }
}

void Render_engine::generate_normal_segment_line(
    My_Type::Vector3f *dev_raw_aligned_points, My_Type::Vector3f *dev_normals,
    NormalsSource normals_source) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Generate normal vector segment lines (for visualization)
  {
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->scene_depth_size.width / (float)thread_rect.x;
    block_rect.y = this->scene_depth_size.height / (float)thread_rect.y;
    block_rect.z = 1;
    switch (normals_source) {
    case NormalsSource::DEPTH_NORMAL: {
      generate_line_segment_CUDA(
          block_rect, thread_rect, dev_raw_aligned_points, dev_normals,
          this->dev_current_hierarchy_normal_to_draw.data_ptrs[0]);
      // CUDA_CKECK_KERNEL;
      //
      checkCudaErrors(
          cudaMemcpy(this->current_hierarchy_normal_to_draw.data_ptrs[0],
                     this->dev_current_hierarchy_normal_to_draw.data_ptrs[0],
                     this->depth_size.width * this->depth_size.height *
                         sizeof(My_Type::Line_segment),
                     cudaMemcpyDeviceToHost));
      break;
    }
    case NormalsSource::MODEL_NORMAL: {
      generate_line_segment_CUDA(
          block_rect, thread_rect, dev_raw_aligned_points, dev_normals,
          this->dev_model_hierarchy_normal_to_draw.data_ptrs[0]);
      // CUDA_CKECK_KERNEL;
      //
      checkCudaErrors(
          cudaMemcpy(this->model_hierarchy_normal_to_draw.data_ptrs[0],
                     this->dev_model_hierarchy_normal_to_draw.data_ptrs[0],
                     this->depth_size.width * this->depth_size.height *
                         sizeof(My_Type::Line_segment),
                     cudaMemcpyDeviceToHost));
      break;
    }
    default:
      break;
    }
  }
}

//
void Render_engine::generate_voxel_block_lines(HashEntry *dev_entries,
                                               int number_of_entries) {
  //
  checkCudaErrors(cudaMemcpy(this->enties_buffer, dev_entries,
                             number_of_entries * sizeof(HashEntry),
                             cudaMemcpyDeviceToHost));

  //
  std::vector<HashEntry> allocated_entries;
  this->number_of_blocks = 0;
  for (int entry_id = 0; entry_id < number_of_entries; entry_id++) {
    HashEntry temp_entry = this->enties_buffer[entry_id];
    if (temp_entry.ptr < 0)
      continue;

    allocated_entries.push_back(temp_entry);
    this->number_of_blocks++;
  }
  // Allocate memory
  this->voxel_block_lines = (My_Type::Vector3f *)malloc(
      this->number_of_blocks * sizeof(My_Type::Vector3f) * 24);

  //
  for (int block_id = 0; block_id < allocated_entries.size(); block_id++) {
    HashEntry temp_entry = allocated_entries[block_id];

    My_Type::Vector3f point_list[8];
    point_list[0] = My_Type::Vector3f(
        temp_entry.position[0], temp_entry.position[1], temp_entry.position[2]);
    float block_width = VOXEL_BLOCK_WDITH * VOXEL_SIZE;
    point_list[0] *= block_width;
    point_list[1] = point_list[0] + My_Type::Vector3f(block_width, 0, 0);
    point_list[2] = point_list[0] + My_Type::Vector3f(0, block_width, 0);
    point_list[3] =
        point_list[0] + My_Type::Vector3f(block_width, block_width, 0);
    point_list[4] = point_list[0] + My_Type::Vector3f(0, 0, block_width);
    point_list[5] =
        point_list[0] + My_Type::Vector3f(block_width, 0, block_width);
    point_list[6] =
        point_list[0] + My_Type::Vector3f(0, block_width, block_width);
    point_list[7] = point_list[0] +
                    My_Type::Vector3f(block_width, block_width, block_width);

    //
    int vertex_index = block_id * 24;
    // X direction
    this->voxel_block_lines[vertex_index + 0] = point_list[0];
    this->voxel_block_lines[vertex_index + 1] = point_list[1];
    this->voxel_block_lines[vertex_index + 2] = point_list[2];
    this->voxel_block_lines[vertex_index + 3] = point_list[3];
    this->voxel_block_lines[vertex_index + 4] = point_list[4];
    this->voxel_block_lines[vertex_index + 5] = point_list[5];
    this->voxel_block_lines[vertex_index + 6] = point_list[6];
    this->voxel_block_lines[vertex_index + 7] = point_list[7];
    // Y direction
    this->voxel_block_lines[vertex_index + 8] = point_list[0];
    this->voxel_block_lines[vertex_index + 9] = point_list[2];
    this->voxel_block_lines[vertex_index + 10] = point_list[1];
    this->voxel_block_lines[vertex_index + 11] = point_list[3];
    this->voxel_block_lines[vertex_index + 12] = point_list[4];
    this->voxel_block_lines[vertex_index + 13] = point_list[6];
    this->voxel_block_lines[vertex_index + 14] = point_list[5];
    this->voxel_block_lines[vertex_index + 15] = point_list[7];
    // Z direction
    this->voxel_block_lines[vertex_index + 16] = point_list[0];
    this->voxel_block_lines[vertex_index + 17] = point_list[4];
    this->voxel_block_lines[vertex_index + 18] = point_list[1];
    this->voxel_block_lines[vertex_index + 19] = point_list[5];
    this->voxel_block_lines[vertex_index + 20] = point_list[2];
    this->voxel_block_lines[vertex_index + 21] = point_list[6];
    this->voxel_block_lines[vertex_index + 22] = point_list[3];
    this->voxel_block_lines[vertex_index + 23] = point_list[7];
  }
}
