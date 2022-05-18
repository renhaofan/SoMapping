

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "Voxel_map.h"
// CUDA KernelFunctions header
#include "Voxel_map_KernelFunc.cuh"
//#include "Render_KernelFunc.cuh"

//
#include "OurLib/my_math_functions.h"
#include "SLAM_system/SLAM_system_settings.h"

// C/C++ IO
#include <iostream>
#include <stdio.h>
using namespace std;

//
Voxel_map::Voxel_map() {
  //
  this->number_of_blocks = 0;
  out_of_block = false;
  //
  // this->submap_pose.mat.setIdentity();
}
Voxel_map::~Voxel_map() {
  // printf("Call ~Voxel_map()\n");

  // CUDA
  // collision counter
  checkCudaErrors(cudaFree(this->dev_collision_counter));
  //       Block
  checkCudaErrors(cudaFree(this->dev_number_of_blocks));
  //         Entry
  checkCudaErrors(cudaFree(this->dev_number_of_visible_blocks));
  //
  checkCudaErrors(cudaFree(this->dev_number_of_relative_blocks));
  //
  checkCudaErrors(cudaFree(this->dev_min_depth));
  checkCudaErrors(cudaFree(this->dev_max_depth));
  // Entry	    Entry
  checkCudaErrors(cudaFree(this->dev_entrise));
  checkCudaErrors(cudaFree(this->dev_visible_list));
  checkCudaErrors(cudaFree(this->dev_relative_list));
  //
  checkCudaErrors(cudaFree(this->dev_allocate_flag));
  //       Enty
  checkCudaErrors(cudaFree(this->dev_visible_flag));
  //
  checkCudaErrors(cudaFree(this->dev_ordered_position));
  //
  checkCudaErrors(cudaFree(this->dev_voxel_block_array));

  //
  checkCudaErrors(cudaFree(this->dev_raycast_range_map));
  checkCudaErrors(cudaFree(this->dev_scene_range_map));

  //
  checkCudaErrors(cudaFree(this->dev_raycast_points));
  checkCudaErrors(cudaFree(this->dev_scene_points));
  //
  checkCudaErrors(cudaFree(this->dev_raycast_normal));
  checkCudaErrors(cudaFree(this->dev_scene_normals));
  //
  checkCudaErrors(cudaFree(this->dev_raycast_weight));
  checkCudaErrors(cudaFree(this->dev_scene_weight));
  //
  checkCudaErrors(cudaFree(this->dev_raycast_plane_label));
  checkCudaErrors(cudaFree(this->dev_scene_plane_label));
  //
  checkCudaErrors(cudaFree(this->dev_voxel_block_position));
  //
  // checkCudaErrors(cudaFree(this->dev_scene_color));

  //
  checkCudaErrors(cudaFree(this->dev_current_weight_center));
  checkCudaErrors(cudaFree(this->dev_map_weight_center));

  // CPU
  free(this->visible_list);
  free(this->allocate_flag);
  free(this->ordered_position);
  free(this->raycast_points);
  free(this->raycast_normal);
  free(this->voxel_block_position);
  free(this->scene_points);
  // free(this->scene_color);
}

//
void Voxel_map::init_Voxel_map(My_Type::Vector2i aligned_depth_size,
                               int voxel_block_num) {
  //
  this->max_voxel_block_number = voxel_block_num;

  // Get tracking image size
  this->raycast_depth_width = aligned_depth_size.width;
  this->raycast_depth_height = aligned_depth_size.height;
  // Compute raytcast range map size ( '1 + ...' for boundary)
  this->raycast_range_map_width =
      1 +
      (int)ceilf(
          (float)this->raycast_depth_width /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);
  this->raycast_range_map_height =
      1 +
      (int)ceilf(
          (float)this->raycast_depth_height /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);

  // Get scene image size
  this->scene_depth_width = aligned_depth_size.width;
  this->scene_depth_height = aligned_depth_size.height;
  // Compute scene range map size ( '1 + ...' for boundary)
  this->scene_range_map_width =
      1 +
      (int)ceilf(
          (float)this->scene_depth_width /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);
  this->scene_range_map_height =
      1 +
      (int)ceilf(
          (float)this->scene_depth_height /
          (float)SLAM_system_settings::instance()->raycast_range_patch_width);

  // collision counter
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_collision_counter), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_blocks), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_visible_blocks), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_relative_blocks), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_min_depth), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_max_depth), sizeof(int)));
  // entry
  checkCudaErrors(cudaMalloc((void **)&(this->dev_entrise),
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_visible_list),
                             VISIBLE_LIST_LENGTH * sizeof(HashEntry)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_relative_list),
                             this->max_voxel_block_number * sizeof(HashEntry)));
  // flag
  checkCudaErrors(cudaMalloc((void **)&(this->dev_allocate_flag),
                             ORDERED_TABLE_LENGTH * sizeof(char)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_visible_flag),
                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(char)));
  // position
  checkCudaErrors(cudaMalloc((void **)&(this->dev_ordered_position),
                             ORDERED_TABLE_LENGTH * sizeof(My_Type::Vector3i)));
  // Voxel Block Array
  checkCudaErrors(cudaMalloc((void **)&(this->dev_voxel_block_array),
                             this->max_voxel_block_number * VOXEL_BLOCK_SIZE *
                                 sizeof(Voxel_f)));

  // Raycast Range
  checkCudaErrors(cudaMalloc((void **)&(this->dev_raycast_range_map),
                             this->raycast_range_map_width *
                                 this->raycast_range_map_height *
                                 sizeof(My_Type::Vector2f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_range_map),
                 this->scene_range_map_width * this->scene_range_map_height *
                     sizeof(My_Type::Vector2f)));

  // Raycast Points
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_raycast_points),
                 this->raycast_depth_width * this->raycast_depth_height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_points),
                 this->scene_depth_width * this->scene_depth_height *
                     sizeof(My_Type::Vector3f)));
  // Raycast normal
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_raycast_normal),
                 this->raycast_depth_width * this->raycast_depth_height *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_scene_normals),
                 this->scene_depth_width * this->scene_depth_height *
                     sizeof(My_Type::Vector3f)));
  // Raycast weight
  checkCudaErrors(cudaMalloc((void **)&(this->dev_raycast_weight),
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_weight),
                             this->scene_depth_width *
                                 this->scene_depth_height * sizeof(int)));
  // Raycast Plane label
  checkCudaErrors(cudaMalloc((void **)&(this->dev_raycast_plane_label),
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_plane_label),
                             this->scene_depth_width *
                                 this->scene_depth_height * sizeof(int)));

  // for view
  // Voxel block position
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_voxel_block_position),
                 this->max_voxel_block_number * sizeof(My_Type::Vector3i)));
  // Raycast Scene color
  // checkCudaErrors(cudaMalloc((void **)&(this->dev_scene_color),
  // this->scene_depth_width * this->scene_depth_height *
  // sizeof(My_Type::Vector4uc)));
  //
  checkCudaErrors(cudaMalloc((void **)&(this->dev_current_weight_center),
                             sizeof(My_Type::Vector3f)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_map_weight_center),
                             sizeof(My_Type::Vector3f)));

  //
  // collision counter
  checkCudaErrors(cudaMemset(this->dev_collision_counter, 0x00, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_number_of_blocks, 0x00, sizeof(int)));
  checkCudaErrors(
      cudaMemset(this->dev_number_of_visible_blocks, 0x00, sizeof(int)));
  // entry
  checkCudaErrors(cudaMemset(this->dev_entrise, 0xF7,
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));
  // voxel      0 （   0）
  checkCudaErrors(cudaMemset(this->dev_voxel_block_array, 0x00,
                             this->max_voxel_block_number * VOXEL_BLOCK_SIZE *
                                 sizeof(Voxel_f)));
  //
  checkCudaErrors(cudaMemset(this->dev_raycast_plane_label, 0x00,
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(int)));

  // CPU
  this->allocate_flag = (char *)malloc(ORDERED_TABLE_LENGTH * sizeof(char));
  this->ordered_position = (My_Type::Vector3i *)malloc(
      ORDERED_TABLE_LENGTH * sizeof(My_Type::Vector3i));
  this->visible_list =
      (HashEntry *)malloc(VISIBLE_LIST_LENGTH * sizeof(HashEntry));
  //
  this->raycast_range_map = (My_Type::Vector2f *)malloc(
      this->raycast_range_map_width * this->raycast_range_map_height *
      sizeof(My_Type::Vector2f));
  this->scene_range_map = (My_Type::Vector2f *)malloc(
      this->scene_range_map_width * this->scene_range_map_height *
      sizeof(My_Type::Vector2f));
  //
  this->raycast_points = (My_Type::Vector3f *)malloc(
      this->raycast_depth_width * this->raycast_depth_height *
      sizeof(My_Type::Vector3f));
  this->raycast_normal = (My_Type::Vector3f *)malloc(
      this->raycast_depth_width * this->raycast_depth_height *
      sizeof(My_Type::Vector3f));
  this->voxel_block_position = (My_Type::Vector3i *)malloc(
      this->max_voxel_block_number * sizeof(My_Type::Vector3i));
  // for view
  this->scene_points = (My_Type::Vector3f *)malloc(this->scene_depth_width *
                                                   this->scene_depth_height *
                                                   sizeof(My_Type::Vector3f));
  // this->scene_color = (My_Type::Vector4uc *)malloc(this->scene_depth_width *
  // this->scene_depth_height * sizeof(My_Type::Vector4uc));
}

//   entry  block
int Voxel_map::allocate_voxel_block(My_Type::Vector3f *dev_current_points,
                                    Eigen::Matrix4f camera_pose,
                                    Eigen::Matrix4f sumap_pose) {
  //   Voxel Block
  if (this->out_of_block)
    return this->number_of_blocks;

  //
  // this->camera_pose_in_submap.mat = this->submap_pose.mat.inverse() *
  // camera_pose;
  this->camera_pose_in_submap.mat = sumap_pose.inverse() * camera_pose;
  this->camera_pose_in_submap.synchronize_to_GPU();

  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(cudaMemcpy(&(this->number_of_blocks),
                             this->dev_number_of_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));
  int last_block_num = this->number_of_blocks;
  int block_inc = 1;
  while (block_inc > 0) {
    //
    checkCudaErrors(cudaMemset(this->dev_allocate_flag, 0x00,
                               ORDERED_TABLE_LENGTH * sizeof(char)));

    // Set CUDA kernel lunch paramenters
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->raycast_depth_width / thread_rect.x;
    block_rect.y = this->raycast_depth_height / thread_rect.y;
    block_rect.z = 1;

    // Lunch CUDA kernel function
    build_entry_flag_CUDA(block_rect, thread_rect, dev_current_points,
                          this->camera_pose_in_submap.dev_mat,
                          this->dev_entrise, this->dev_allocate_flag,
                          this->dev_ordered_position);
    //		CUDA_CKECK_KERNEL;

    // VoxelBlock Entry
    thread_rect.x = 512;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = ORDERED_TABLE_LENGTH / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;

    // Lunch CUDA kernel function
    allocate_by_flag_CUDA(
        block_rect, thread_rect, this->dev_entrise, this->dev_allocate_flag,
        this->dev_ordered_position, this->dev_collision_counter,
        this->dev_number_of_blocks, this->dev_voxel_block_position);
    //		CUDA_CKECK_KERNEL;

    // Block
    checkCudaErrors(cudaMemcpy(&(this->number_of_blocks),
                               this->dev_number_of_blocks, sizeof(int),
                               cudaMemcpyDeviceToHost));
    block_inc = this->number_of_blocks - last_block_num;
    last_block_num = this->number_of_blocks;
    // cout << "block_inc = " << block_inc << endl;
    if (this->max_voxel_block_number < this->number_of_blocks) {
      printf("Map use out of voxel block! : %d < %d\r\n",
             this->max_voxel_block_number, this->number_of_blocks);
      this->out_of_block = true;
      return this->number_of_blocks;
    }
  }

  // visible
  checkCudaErrors(
      cudaMemset(this->dev_visible_flag, 0x00,
                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(char)));

  // Set CUDA kernel lunch paramenters
  thread_rect.x = SLAM_system_settings::instance()->raycast_range_patch_width;
  thread_rect.y = SLAM_system_settings::instance()->raycast_range_patch_width;
  thread_rect.z = 1;
  block_rect.x = this->raycast_depth_width / thread_rect.x;
  block_rect.y = this->raycast_depth_height / thread_rect.y;
  block_rect.z = 1;

  // Lunch CUDA kernel function
  build_visible_flag_CUDA(block_rect, thread_rect, dev_current_points,
                          this->camera_pose_in_submap.dev_mat,
                          this->dev_entrise, this->dev_visible_flag);
  // CUDA_CKECK_KERNEL;

  checkCudaErrors(
      cudaMemset(this->dev_number_of_visible_blocks, 0x00, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_min_depth, 0x7F, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_max_depth, 0x00, sizeof(int)));

  thread_rect.x = 512;
  thread_rect.y = 1;
  thread_rect.z = 1;
  block_rect.x = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) / thread_rect.x;
  block_rect.y = 1;
  block_rect.z = 1;

  // Lunch CUDA kernel function
  build_visible_list_CUDA(block_rect, thread_rect, this->dev_entrise,
                          this->dev_visible_list, this->dev_visible_flag,
                          this->dev_number_of_visible_blocks,
                          this->camera_pose_in_submap.dev_mat_inv,
                          this->dev_min_depth, this->dev_max_depth);
  // CUDA_CKECK_KERNEL;

  return this->number_of_blocks;
}

//   entry  block
void Voxel_map::fusion_SDF_to_voxel(My_Type::Vector3f *dev_current_points,
                                    My_Type::Vector3f *dev_current_normal,
                                    Eigen::Matrix4f camera_pose,
                                    Eigen::Matrix4f submap_pose) {
  //
  // this->camera_pose_in_submap.mat = this->submap_pose.mat.inverse() *
  // camera_pose;
  this->camera_pose_in_submap.mat = submap_pose.inverse() * camera_pose;
  this->camera_pose_in_submap.synchronize_to_GPU();

  // VisibleList
  checkCudaErrors(cudaMemcpy(&(this->number_of_visible_blocks),
                             this->dev_number_of_visible_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));

  // VoxelBlock
  dim3 block_rect(1, 1, 1),
      thread_rect(VOXEL_BLOCK_WDITH, VOXEL_BLOCK_WDITH, VOXEL_BLOCK_WDITH);
  block_rect.x = this->number_of_visible_blocks - 1;

  // Lunch CUDA kernel function
  if (true) {
    prj_normal_fusion_sdf_CUDA(
        block_rect, thread_rect, dev_current_points, dev_current_normal,
        this->camera_pose_in_submap.dev_mat_inv,
        SLAM_system_settings::instance()->sensor_params,
        this->raycast_depth_width, this->raycast_depth_height,
        this->dev_visible_list, this->dev_voxel_block_array);
  } else {
    prj_fusion_sdf_CUDA(block_rect, thread_rect, dev_current_points,
                        this->camera_pose_in_submap.dev_mat_inv,
                        SLAM_system_settings::instance()->sensor_params,
                        this->raycast_depth_width, this->raycast_depth_height,
                        this->dev_visible_list, this->dev_voxel_block_array);
  }
  // CUDA_CKECK_KERNEL;
}
//
void Voxel_map::fusion_plane_label_to_voxel(
    My_Type::Vector3f *dev_current_points, int *plane_img,
    Eigen::Matrix4f camera_pose, Eigen::Matrix4f submap_pose) {

  //
  // this->camera_pose_in_submap.mat = this->submap_pose.mat.inverse() * pose;
  this->camera_pose_in_submap.mat = submap_pose.inverse() * camera_pose;
  this->camera_pose_in_submap.synchronize_to_GPU();

  //   VisibleList
  checkCudaErrors(cudaMemcpy(&(this->number_of_visible_blocks),
                             this->dev_number_of_visible_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));

  // VoxelBlock
  dim3 block_rect(1, 1, 1),
      thread_rect(VOXEL_BLOCK_WDITH, VOXEL_BLOCK_WDITH, VOXEL_BLOCK_WDITH);
  block_rect.x = this->number_of_visible_blocks;

  // Lunch CUDA kernel function
  prj_fusion_plane_label_CUDA(
      block_rect, thread_rect, dev_current_points,
      this->camera_pose_in_submap.dev_mat_inv,
      SLAM_system_settings::instance()->sensor_params,
      this->raycast_depth_width, this->raycast_depth_height,
      this->dev_visible_list, this->dev_voxel_block_array, plane_img);
  // CUDA_CKECK_KERNEL;
}

// Raycast
void Voxel_map::raycast_by_pose(Eigen::Matrix4f camera_pose, RaycastMode mode) {
  //
  // this->camera_pose_in_submap.mat = this->submap_pose.mat.inverse() *
  // camera_pose;
  this->camera_pose_in_submap.mat = camera_pose;
  this->camera_pose_in_submap.synchronize_to_GPU();

  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Reduce range			// TODO: dev_voxel_block_position
  {
    checkCudaErrors(cudaMemcpy(&(this->number_of_blocks),
                               this->dev_number_of_blocks, sizeof(int),
                               cudaMemcpyDeviceToHost));
    // cout << "number_of_blocks = \t" << this->number_of_blocks << endl;

    checkCudaErrors(cudaMemset(this->dev_min_depth, 0x7F, sizeof(int)));
    checkCudaErrors(cudaMemset(this->dev_max_depth, 0x00, sizeof(int)));

    // Set CUDA kernel lunch paramenters
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = ceil_by_stride(this->number_of_blocks, thread_rect.x);
    block_rect.y = 1;
    block_rect.z = 1;

    // Lunch CUDA kernel function
    reduce_range_CUDA(block_rect, thread_rect, this->dev_visible_list,
                      this->number_of_blocks, this->dev_min_depth,
                      this->dev_max_depth,
                      this->camera_pose_in_submap.dev_mat_inv);
    // CUDA_CKECK_KERNEL;
  }

  checkCudaErrors(cudaMemcpy(&(this->min_depth), this->dev_min_depth,
                             sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&(this->max_depth), this->dev_max_depth,
                             sizeof(int), cudaMemcpyDeviceToHost));
  // cout << this->min_depth << "," << this->max_depth << endl;

  //
  switch (mode) {
  case RaycastMode::RAYCSAT_FOR_TRACKING: {
    //-------------- Get range map
    //       Block   ，    Patch（16x16） Range
    checkCudaErrors(cudaMemset(this->dev_raycast_range_map, 0x00,
                               this->raycast_range_map_width *
                                   this->raycast_range_map_height *
                                   sizeof(My_Type::Vector2f)));

    // ---------------------------- To Do : validate memory access safety
    //  PatchSize   Raycast        Raycast   Patch Range
    // Set CUDA kernel lunch paramenters
    thread_rect.x = this->raycast_range_map_width - 1;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = this->raycast_range_map_height - 1;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch CUDA kernel function
    if (true) {
      raycast_get_range_4corner_CUDA(
          block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
          SLAM_system_settings::instance()->sensor_params,
          SLAM_system_settings::instance()->raycast_range_patch_width,
          this->dev_entrise, this->dev_min_depth, this->dev_max_depth,
          this->dev_raycast_range_map);
    } else {
      raycast_get_range_CUDA(
          block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
          this->dev_entrise, SLAM_system_settings::instance()->sensor_params,
          SLAM_system_settings::instance()->raycast_range_patch_width,
          this->dev_min_depth, this->dev_max_depth,
          this->dev_raycast_range_map);
    }
    // CUDA_CKECK_KERNEL;

    // ----------------------- Raycast
    thread_rect.x = SLAM_system_settings::instance()->raycast_range_patch_width;
    thread_rect.y = SLAM_system_settings::instance()->raycast_range_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->raycast_depth_width / thread_rect.x;
    block_rect.y = this->raycast_depth_height / thread_rect.y;
    block_rect.z = 1;

    checkCudaErrors(
        cudaMemset(this->dev_raycast_points, 0x00,
                   this->raycast_depth_width * this->raycast_depth_height *
                       sizeof(My_Type::Vector3f)));
    checkCudaErrors(
        cudaMemset(this->dev_raycast_normal, 0x00,
                   this->raycast_depth_width * this->raycast_depth_height *
                       sizeof(My_Type::Vector3f)));
    raycast_byStep_CUDA(
        block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
        SLAM_system_settings::instance()->sensor_params,
        SLAM_system_settings::instance()->raycast_range_patch_width,
        this->dev_entrise, this->dev_voxel_block_array,
        this->dev_raycast_range_map, this->dev_raycast_points,
        this->dev_raycast_normal, this->dev_raycast_plane_label,
        this->dev_raycast_weight);
    // CUDA_CKECK_KERNEL;

    break;
  }
  case RaycastMode::RAYCAST_FOR_VIEW: {

    //-------------- Get range map
    //
    checkCudaErrors(
        cudaMemset(this->dev_scene_range_map, 0x00,
                   this->scene_range_map_width * this->scene_range_map_height *
                       sizeof(My_Type::Vector2f)));

    //
    //
    thread_rect.x = this->scene_range_map_width - 1;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = this->scene_range_map_height - 1;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch CUDA kernel function
    if (true) {
      raycast_get_range_4corner_CUDA(
          block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
          SLAM_system_settings::instance()->sensor_params,
          SLAM_system_settings::instance()->raycast_range_patch_width,
          this->dev_entrise, this->dev_min_depth, this->dev_max_depth,
          this->dev_scene_range_map);
    } else {
      raycast_get_range_CUDA(
          block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
          this->dev_entrise, SLAM_system_settings::instance()->sensor_params,
          SLAM_system_settings::instance()->raycast_range_patch_width,
          this->dev_min_depth, this->dev_max_depth, this->dev_scene_range_map);
    }
    // CUDA_CKECK_KERNEL;

    checkCudaErrors(cudaMemcpy(&(this->min_depth), this->dev_min_depth,
                               sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&(this->max_depth), this->dev_max_depth,
                               sizeof(int), cudaMemcpyDeviceToHost));

    // ----------------------- Raycast
    thread_rect.x = SLAM_system_settings::instance()->raycast_range_patch_width;
    thread_rect.y = SLAM_system_settings::instance()->raycast_range_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->scene_depth_width / thread_rect.x;
    block_rect.y = this->scene_depth_height / thread_rect.y;
    block_rect.z = 1;

    checkCudaErrors(
        cudaMemset(this->dev_scene_points, 0x00,
                   this->scene_depth_width * this->scene_depth_height *
                       sizeof(My_Type::Vector3f)));
    checkCudaErrors(
        cudaMemset(this->dev_scene_normals, 0x00,
                   this->scene_depth_width * this->scene_depth_height *
                       sizeof(My_Type::Vector3f)));
    raycast_byStep_CUDA(
        block_rect, thread_rect, this->camera_pose_in_submap.dev_mat,
        SLAM_system_settings::instance()->sensor_params,
        SLAM_system_settings::instance()->raycast_range_patch_width,
        this->dev_entrise, this->dev_voxel_block_array,
        this->dev_scene_range_map, this->dev_scene_points,
        this->dev_scene_normals, this->dev_scene_plane_label,
        this->dev_scene_weight);
    //			CUDA_CKECK_KERNEL;
    //
    break;
  }
  default: {
    printf("Illegal mode!\r\n");
    break;
  }
  }
}

//
void Voxel_map::merge_with_voxel_map(const Voxel_map &fragment_map,
                                     int *dev_plane_global_index) {}

void Voxel_map::update_from_last_voxel_map(
    My_pose &camera_pose, My_Type::Vector3f *dev_current_points,
    HashEntry *dev_entries, Voxel_f *dev_voxel_array) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  camera_pose.synchronize_to_GPU();

  int number_of_visible_enteies = 1;
  int *dev_number_of_visible_enteies = nullptr;
  char *dev_temp_visible_flag = nullptr;
  HashEntry *dev_temp_visible_entries = nullptr;

  // Allocation
  checkCudaErrors(
      cudaMalloc((void **)&(dev_number_of_visible_enteies), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(dev_temp_visible_flag),
                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(char)));
  checkCudaErrors(cudaMalloc((void **)&(dev_temp_visible_entries),
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));

  for (int i = 0; (i < 20 && number_of_visible_enteies != 0); i++) {
    // Init data
    checkCudaErrors(
        cudaMemset(dev_number_of_visible_enteies, 0x00, sizeof(int)));
    checkCudaErrors(cudaMemset(dev_temp_visible_flag, 0x00,
                               (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                   sizeof(char)));

    // Find visible entries
    {
      thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
      thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
      thread_rect.z = 1;
      block_rect.x =
          SLAM_system_settings::instance()->aligned_depth_size.width /
          thread_rect.x;
      block_rect.y =
          SLAM_system_settings::instance()->aligned_depth_size.height /
          thread_rect.y;
      block_rect.z = 1;
      build_visible_flag_CUDA(block_rect, thread_rect, dev_current_points,
                              camera_pose.dev_mat, dev_entries,
                              this->dev_entrise, dev_temp_visible_flag);
      // CUDA_CKECK_KERNEL;

      //
      thread_rect.x = 512;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) / thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      build_visible_list_CUDA(block_rect, thread_rect, dev_entries,
                              dev_temp_visible_entries, dev_temp_visible_flag,
                              dev_number_of_visible_enteies);
      // CUDA_CKECK_KERNEL;
    }

    // Voxel block allocation (this map)
    {
      checkCudaErrors(cudaMemcpy(&number_of_visible_enteies,
                                 dev_number_of_visible_enteies, sizeof(int),
                                 cudaMemcpyDeviceToHost));
      // printf("number_of_visible_enteies = %d\n", number_of_visible_enteies);

      thread_rect.x = VOXEL_BLOCK_WDITH;
      thread_rect.y = VOXEL_BLOCK_WDITH;
      thread_rect.z = VOXEL_BLOCK_WDITH;
      block_rect.x = number_of_visible_enteies;
      block_rect.y = 1;
      block_rect.z = 1;
      update_voxel_from_visible_blocks_CUDA(
          block_rect, thread_rect, dev_temp_visible_entries, dev_voxel_array,
          this->dev_entrise, this->dev_voxel_block_array,
          this->dev_collision_counter, this->dev_number_of_blocks);
      // CUDA_CKECK_KERNEL;

      // printf("number_of_visible_enteies = %d\n", number_of_visible_enteies);
    }
  }

  // Release
  checkCudaErrors(cudaFree(dev_number_of_visible_enteies));
  checkCudaErrors(cudaFree(dev_temp_visible_flag));
  checkCudaErrors(cudaFree(dev_temp_visible_entries));
}

//
void Voxel_map::clear_Voxel_map() {
  //
  // collision counter
  checkCudaErrors(cudaMemset(this->dev_collision_counter, 0x00, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_number_of_blocks, 0x00, sizeof(int)));
  checkCudaErrors(
      cudaMemset(this->dev_number_of_visible_blocks, 0x00, sizeof(int)));
  // entry           （ptr  ）
  checkCudaErrors(cudaMemset(this->dev_entrise, 0xF7,
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));
  // voxel      0 （   0）
  checkCudaErrors(cudaMemset(this->dev_voxel_block_array, 0x00,
                             this->max_voxel_block_number * VOXEL_BLOCK_SIZE *
                                 sizeof(Voxel_f)));
  //
  checkCudaErrors(cudaMemset(this->dev_raycast_plane_label, 0x00,
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(int)));
}

//
void Voxel_map::compress_voxel_map() {
  // this->number_of_blocks
  Voxel_f *dev_swap_pointer = nullptr;

  //
  printf("total voxel blocks = %d\n", this->number_of_blocks);

  // Re-allocate
  checkCudaErrors(
      cudaMalloc((void **)&(dev_swap_pointer),
                 this->number_of_blocks * VOXEL_BLOCK_SIZE * sizeof(Voxel_f)));
  // Copy data
  checkCudaErrors(
      cudaMemcpy(dev_swap_pointer, this->dev_voxel_block_array,
                 this->number_of_blocks * VOXEL_BLOCK_SIZE * sizeof(Voxel_f),
                 cudaMemcpyDeviceToDevice));
  // Release old VBA
  checkCudaErrors(cudaFree(this->dev_voxel_block_array));
  // Swap buffer pointer to swap pointer
  this->dev_voxel_block_array = dev_swap_pointer;
}

//
void Voxel_map::release_voxel_map() {
  checkCudaErrors(cudaFree(this->dev_collision_counter));
  checkCudaErrors(cudaFree(this->dev_number_of_blocks));
  checkCudaErrors(cudaFree(this->dev_number_of_visible_blocks));
  checkCudaErrors(cudaFree(this->dev_number_of_relative_blocks));
  checkCudaErrors(cudaFree(this->dev_min_depth));
  checkCudaErrors(cudaFree(this->dev_max_depth));
  checkCudaErrors(cudaFree(this->dev_entrise));
  checkCudaErrors(cudaFree(this->dev_visible_list));
  checkCudaErrors(cudaFree(this->dev_relative_list));
  checkCudaErrors(cudaFree(this->dev_allocate_flag));
  checkCudaErrors(cudaFree(this->dev_visible_flag));
  checkCudaErrors(cudaFree(this->dev_ordered_position));
  checkCudaErrors(cudaFree(this->dev_voxel_block_array));
  checkCudaErrors(cudaFree(this->dev_raycast_range_map));
  checkCudaErrors(cudaFree(this->dev_scene_range_map));
  checkCudaErrors(cudaFree(this->dev_raycast_points));
  checkCudaErrors(cudaFree(this->dev_scene_points));
  checkCudaErrors(cudaFree(this->dev_raycast_normal));
  checkCudaErrors(cudaFree(this->dev_scene_normals));
  checkCudaErrors(cudaFree(this->dev_raycast_weight));
  checkCudaErrors(cudaFree(this->dev_scene_weight));
  checkCudaErrors(cudaFree(this->dev_raycast_plane_label));
  checkCudaErrors(cudaFree(this->dev_scene_plane_label));
  checkCudaErrors(cudaFree(this->dev_voxel_block_position));
  checkCudaErrors(cudaFree(this->dev_current_weight_center));
  checkCudaErrors(cudaFree(this->dev_map_weight_center));

  this->dev_collision_counter = nullptr;
  this->dev_number_of_blocks = nullptr;
  this->dev_number_of_visible_blocks = nullptr;
  this->dev_number_of_relative_blocks = nullptr;
  this->dev_min_depth = nullptr;
  this->dev_max_depth = nullptr;
  this->dev_entrise = nullptr;
  this->dev_visible_list = nullptr;
  this->dev_relative_list = nullptr;
  this->dev_allocate_flag = nullptr;
  this->dev_visible_flag = nullptr;
  this->dev_ordered_position = nullptr;
  this->dev_voxel_block_array = nullptr;
  this->dev_raycast_range_map = nullptr;
  this->dev_scene_range_map = nullptr;
  this->dev_raycast_points = nullptr;
  this->dev_scene_points = nullptr;
  this->dev_raycast_normal = nullptr;
  this->dev_scene_normals = nullptr;
  this->dev_raycast_weight = nullptr;
  this->dev_scene_weight = nullptr;
  this->dev_raycast_plane_label = nullptr;
  this->dev_scene_plane_label = nullptr;
  this->dev_voxel_block_position = nullptr;
  this->dev_current_weight_center = nullptr;
  this->dev_map_weight_center = nullptr;
}

//
//
////
// void Voxel_map::render_map(Eigen::Matrix4f trans_mat, int render_mode)
//{
//	dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);
//
//	// Raycast render informations
//	Eigen::Matrix4f pose = trans_mat.inverse();
//	this->raycast_by_pose(pose, RaycastMode::RAYCAST_FOR_VIEW);
//
//	//
//	checkCudaErrors(cudaMemset(this->dev_scene_color, 0x00,
//this->scene_depth_width * this->scene_depth_height *
//sizeof(My_Type::Vector4uc)));
//
//
//	//
//	// Set CUDA kernel lunch paramenters
//	thread_rect.x =
//SLAM_system_settings::instance()->image_alginment_patch_width; 	thread_rect.y =
//SLAM_system_settings::instance()->image_alginment_patch_width; 	thread_rect.z =
//1; 	block_rect.x = this->scene_depth_width / thread_rect.x; 	block_rect.y =
//this->scene_depth_height / thread_rect.y; 	block_rect.z = 1;
//
//
//	//         Lunch CUDA kernel function
//	switch (render_mode)
//	{
//		case 1:
//		{
//			//
//			render_gypsum_CUDA(block_rect, thread_rect,
//this->dev_scene_normals, this->dev_scene_color); 			break;
//		}
//		case 2:
//		{
//			//
//			render_weight_CUDA(block_rect, thread_rect,
//this->dev_scene_normals, this->dev_scene_weight, this->dev_scene_color);
//			break;
//		}
//		case 3:
//		{
//			//
//			render_plane_label_CUDA(block_rect, thread_rect,
//this->dev_scene_normals, this->dev_scene_plane_label, this->dev_scene_color);
//			break;
//		}
//	default:
//		break;
//	}
//	CUDA_CKECK_KERNEL;
//
//
//	//
//	checkCudaErrors(cudaMemcpy(this->scene_points, this->dev_scene_points,
//this->scene_depth_width * this->scene_depth_height *
//sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));
//	//
//	checkCudaErrors(cudaMemcpy(this->scene_color, this->dev_scene_color,
//this->scene_depth_width * this->scene_depth_height *
//sizeof(My_Type::Vector4uc), cudaMemcpyDeviceToHost));
//
//
//}
//

//    Visible Block List  CPU
void Voxel_map::extract_visible_list_to_CPU() {
  //
  checkCudaErrors(cudaMemcpy(this->allocate_flag, this->dev_allocate_flag,
                             ORDERED_TABLE_LENGTH * sizeof(char),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(this->ordered_position, this->dev_ordered_position,
                             ORDERED_TABLE_LENGTH * sizeof(My_Type::Vector3i),
                             cudaMemcpyDeviceToHost));

  //
  checkCudaErrors(cudaMemcpy(this->raycast_points, this->dev_raycast_points,
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(float) * 3,
                             cudaMemcpyDeviceToHost));

  //
  int collision_counter;
  checkCudaErrors(cudaMemcpy(&collision_counter, this->dev_collision_counter,
                             sizeof(int), cudaMemcpyDeviceToHost));

  //   Block
  checkCudaErrors(cudaMemcpy(&(this->number_of_blocks),
                             this->dev_number_of_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));
  //       Block
  checkCudaErrors(cudaMemcpy(this->voxel_block_position,
                             this->dev_voxel_block_position,
                             this->number_of_blocks * sizeof(My_Type::Vector3i),
                             cudaMemcpyDeviceToHost));

  //
  // checkCudaErrors(cudaMemcpy(&(this->min_depth), this->dev_min_depth,
  // sizeof(int), cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaMemcpy(&(this->max_depth), this->dev_max_depth,
  // sizeof(int), cudaMemcpyDeviceToHost)); printf("min = %d,  max = %d\r\n",
  // this->min_depth, this->max_depth);

  //
  this->allocate_flag_counter = 0;
  for (int i = 0; i < ORDERED_TABLE_LENGTH; i++) {
    //       allocate
    if (this->allocate_flag[i] == NEED_ALLOCATE) {
      this->allocate_flag_counter++;
      printf("%d, %d, %d\r\n", this->ordered_position[i].x,
             this->ordered_position[i].y, this->ordered_position[i].z);
    }
  }
  cout << "allocate_flag_counter = " << this->allocate_flag_counter << endl;

  //   VisibleList
  checkCudaErrors(cudaMemcpy(&(this->number_of_visible_blocks),
                             this->dev_number_of_visible_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));
  // printf("this->number_of_visible_blocks = %d \r\n",
  // this->number_of_visible_blocks);
  if (this->number_of_visible_blocks > 0) {
    checkCudaErrors(
        cudaMemcpy(this->visible_list, this->dev_visible_list,
                   this->number_of_visible_blocks * sizeof(HashEntry),
                   cudaMemcpyDeviceToHost));
  }
}

//   Raycast
void Voxel_map::copy_out_raycast_points() {
  //
  checkCudaErrors(cudaMemcpy(this->raycast_points, this->dev_raycast_points,
                             this->raycast_depth_width *
                                 this->raycast_depth_height * sizeof(float) * 3,
                             cudaMemcpyDeviceToHost));
  //   RaycastRange
  checkCudaErrors(
      cudaMemcpy(this->raycast_range_map, this->dev_raycast_range_map,
                 this->raycast_range_map_width *
                     this->raycast_range_map_height * sizeof(My_Type::Vector2f),
                 cudaMemcpyDeviceToHost));
}

//
void Voxel_map::reduce_Voxel_map_weight_center() {
  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(cudaMemcpy(&(this->number_of_visible_blocks),
                             this->dev_number_of_visible_blocks, sizeof(int),
                             cudaMemcpyDeviceToHost));

  if (this->number_of_visible_blocks > 0) {
    // initiate paramenters
    checkCudaErrors(cudaMemset(this->dev_current_weight_center, 0x00,
                               sizeof(My_Type::Vector3f)));
    //
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(this->number_of_visible_blocks, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    //
    reduce_current_voxel_block_position_CUDA(
        block_rect, thread_rect, this->dev_visible_list,
        this->dev_current_weight_center, this->number_of_visible_blocks);
    // CUDA_CKECK_KERNEL;
    //
    checkCudaErrors(cudaMemcpy(
        &(this->current_weight_center), this->dev_current_weight_center,
        sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));

    //
    this->current_weight_center /= this->number_of_visible_blocks;
  } else {
    this->current_weight_center.x = 0;
    this->current_weight_center.y = 0;
    this->current_weight_center.z = 0;
  }

  if (this->number_of_blocks > 0) {
    // initiate paramenters
    checkCudaErrors(cudaMemset(this->dev_map_weight_center, 0x00,
                               sizeof(My_Type::Vector3f)));
    //
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    //
    reduce_map_voxel_block_position_CUDA(block_rect, thread_rect,
                                         this->dev_entrise,
                                         this->dev_map_weight_center);
    // CUDA_CKECK_KERNEL;
    //
    Eigen::Vector3f map_weight_center_F, map_weight_center_W;
    checkCudaErrors(
        cudaMemcpy(map_weight_center_F.data(), this->dev_map_weight_center,
                   sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));

    //
    map_weight_center_F /= (float)this->number_of_blocks;
    // map_weight_center_W = submap_pose.mat.block(0, 0, 3, 3) *
    // map_weight_center_F + submap_pose.mat.block(0, 3, 3, 1);
    map_weight_center_W = map_weight_center_F;

    //
    this->map_weight_center.x = map_weight_center_W.x();
    this->map_weight_center.y = map_weight_center_W.y();
    this->map_weight_center.z = map_weight_center_W.z();
  } else {
    this->map_weight_center.x = 0;
    this->map_weight_center.y = 0;
    this->map_weight_center.z = 0;
  }
}
