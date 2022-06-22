

#include "Map_engine/Plane_map.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include <float.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include "math.h"
//
#include "Map_engine/Mesh_generator_KernelFunc.cuh"
#include "Map_engine/Plane_map_KernelFunc.cuh"

//
Plane_map::Plane_map() {
  this->plane_list.push_back(Plane_info(0, 0, 0, 0, 0, 0, false, 0));

  //
  checkCudaErrors(cudaMalloc((void **)&this->dev_plane_list,
                             MAX_CURRENT_PLANES * sizeof(Plane_info)));
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_allocated_entries),
      (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(HashEntry)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_entries), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_pixel_block), sizeof(int)));
  //
  checkCudaErrors(cudaMalloc((void **)&(this->dev_pixel_block_need_allocate),
                             ORDERED_TABLE_LENGTH * sizeof(bool)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_pixel_block_buffer),
                             ORDERED_TABLE_LENGTH * sizeof(My_Type::Vector2i)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_plane_entry_excess_counter),
                             sizeof(int)));
  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_allocated_counter), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_allocated_entry_index),
                 (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH) *
                     sizeof(PlaneHashEntry)));

  //
  this->plane_coordinate_list.clear();
  this->plane_coordinate_list.push_back(Plane_coordinate());

  //
  this->number_of_pixel_blocks = 0;
}
Plane_map::~Plane_map() {
  checkCudaErrors(cudaFree(this->dev_plane_list));
  for (int plane_id = 0; plane_id < this->dev_plane_entry_list.size();
       plane_id++) {
    checkCudaErrors(cudaFree(this->dev_plane_entry_list[plane_id]));
  }
  for (int plane_id = 0; plane_id < this->dev_plane_pixel_array_list.size();
       plane_id++) {
    checkCudaErrors(cudaFree(this->dev_plane_pixel_array_list[plane_id]));
  }

  //
  checkCudaErrors(cudaFree(this->dev_allocated_entries));
  checkCudaErrors(cudaFree(this->dev_number_of_entries));
  checkCudaErrors(cudaFree(this->dev_number_of_pixel_block));
  //
  checkCudaErrors(cudaFree(this->dev_pixel_block_need_allocate));
  checkCudaErrors(cudaFree(this->dev_pixel_block_buffer));
  checkCudaErrors(cudaFree(this->dev_plane_entry_excess_counter));
  //
  checkCudaErrors(cudaFree(this->dev_allocated_counter));
  checkCudaErrors(cudaFree(this->dev_allocated_entry_index));

  // For visualization
  free(this->block_vertexs);
  checkCudaErrors(cudaFree(this->dev_block_vertexs));
}

//
void Plane_map::init() {
  // Device
}

//
void Plane_map::update_plane_list(const Plane_info *current_planes,
                                  std::vector<My_Type::Vector2i> &matches) {
  int current_model_plane_number = this->plane_counter;
  for (int match_id = 1; match_id < matches.size(); match_id++) {
    int current_plane_index = matches[match_id].x;
    if (!current_planes[current_plane_index].is_valid) continue;

    int model_plane_index = matches[match_id].y;
    if (model_plane_index >= current_model_plane_number) {
      this->plane_list.push_back(current_planes[current_plane_index]);
      this->plane_list.back().weight = current_planes[current_plane_index].area;
      this->plane_counter++;
    } else if (model_plane_index > 0) {
      My_Type::Vector4f model_params, current_params;
      //
      model_params.x = this->plane_list[model_plane_index].nx;
      model_params.y = this->plane_list[model_plane_index].ny;
      model_params.z = this->plane_list[model_plane_index].nz;
      model_params.w = this->plane_list[model_plane_index].d;
      //
      current_params.x = current_planes[current_plane_index].nx;
      current_params.y = current_planes[current_plane_index].ny;
      current_params.z = current_planes[current_plane_index].nz;
      current_params.w = current_planes[current_plane_index].d;
      //
      float current_weight = current_planes[current_plane_index].area;
      float model_weight = this->plane_list[model_plane_index].weight;
      model_params =
          model_params * model_weight + current_params * current_weight;
      model_params /= (model_weight + current_weight);
      //
      this->plane_list[model_plane_index].nx = model_params.x;
      this->plane_list[model_plane_index].ny = model_params.y;
      this->plane_list[model_plane_index].nz = model_params.z;
      this->plane_list[model_plane_index].d = model_params.w;
      this->plane_list[model_plane_index].weight =
          fmin(model_weight + current_weight, FLT_MAX);
      // printf("current_weight = %f\n", current_weight);
      // printf("model_weight = %f\n", model_weight);
      // printf("%f, %f, %f, %f\n", current_params.x, current_params.y,
      // current_params.z, current_params.w); printf("%f, %f, %f, %f\n",
      // model_params.x, model_params.y, model_params.z, model_params.w);
    } else {
      // invalid plane
    }
  }

  //! Debug
  if (false) {
    //
    for (int plane_id = 0; plane_id < this->plane_counter; plane_id++) {
      printf("%d : (%f, %f, %f), %f ", plane_id, this->plane_list[plane_id].nx,
             this->plane_list[plane_id].ny, this->plane_list[plane_id].nz,
             this->plane_list[plane_id].d);
      if (this->plane_list[plane_id].is_valid) {
        printf("true \n");
      } else {
        printf("false \n");
      }
    }
    printf("\n");
  }
}

//
void Plane_map::generate_plane_map(const HashEntry *dev_entries,
                                   const Voxel_f *dev_voxel_array) {
  //
  {
    //
    for (int i = 0; i < this->dev_plane_entry_list.size(); i++)
      checkCudaErrors(cudaFree(this->dev_plane_entry_list[i]));
    this->dev_plane_entry_list.clear();
    this->dev_plane_entry_list.push_back(nullptr);
    //
    for (int i = 0; i < this->dev_plane_pixel_array_list.size(); i++)
      checkCudaErrors(cudaFree(this->dev_plane_pixel_array_list[i]));
    this->dev_plane_pixel_array_list.clear();
    this->dev_plane_pixel_array_list.push_back(nullptr);

    //
    this->plane_array_length_list.clear();
    this->plane_array_length_list.push_back(0);
  }

  // Update coordinate of model planes
  {
    // Update old plane coordinate
    for (int plane_id = 1; plane_id < this->plane_coordinate_list.size();
         plane_id++) {
      // Update axis-Z
      // Random generate coordinate
      Eigen::Vector3f base_x, base_y, base_z;
      base_z.x() = this->plane_list[plane_id].nx;
      base_z.y() = this->plane_list[plane_id].ny;
      base_z.z() = this->plane_list[plane_id].nz;
      // Re-orthogonalization axis-X
      base_x.x() = this->plane_coordinate_list[plane_id].x_vec.x;
      base_x.y() = this->plane_coordinate_list[plane_id].x_vec.y;
      base_x.z() = this->plane_coordinate_list[plane_id].x_vec.z;
      base_x = base_x.eval() - base_x.eval().dot(base_z) * base_z;
      base_x.normalize();
      // Re-orthogonalization axis-Y
      base_y = base_z.cross(base_x);

      //
      this->plane_coordinate_list[plane_id].x_vec =
          My_Type::Vector3f(base_x.x(), base_x.y(), base_x.z());
      this->plane_coordinate_list[plane_id].y_vec =
          My_Type::Vector3f(base_y.x(), base_y.y(), base_y.z());
      this->plane_coordinate_list[plane_id].z_vec =
          My_Type::Vector3f(base_z.x(), base_z.y(), base_z.z());
    }

    // New plane coordinate
    for (int plane_id = this->plane_coordinate_list.size();
         plane_id < this->plane_counter; plane_id++) {
      // Random generate coordinate
      Eigen::Vector3f base_x, base_y, base_z;
      base_z.x() = this->plane_list[plane_id].nx;
      base_z.y() = this->plane_list[plane_id].ny;
      base_z.z() = this->plane_list[plane_id].nz;
      // std::cout << plane_id << " - base_z : " << base_z.transpose() <<
      // std::endl;
      // Base vector X
      do {
        base_x.setRandom();
        base_x.normalize();
      } while (fabsf(base_x.dot(base_z)) > 0.7);
      base_x = base_x.eval() - base_x.eval().dot(base_z) * base_z;
      base_x.normalize();
      // Base vector Y
      base_y = base_z.cross(base_x);

      //
      Plane_coordinate temp_coordinate;
      temp_coordinate.x_vec.x = base_x.x();
      temp_coordinate.x_vec.y = base_x.y();
      temp_coordinate.x_vec.z = base_x.z();
      temp_coordinate.y_vec.x = base_y.x();
      temp_coordinate.y_vec.y = base_y.y();
      temp_coordinate.y_vec.z = base_y.z();
      temp_coordinate.z_vec.x = base_z.x();
      temp_coordinate.z_vec.y = base_z.y();
      temp_coordinate.z_vec.z = base_z.z();
      this->plane_coordinate_list.push_back(temp_coordinate);
    }
  }

  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(cudaMemset(this->dev_number_of_entries, 0x00, sizeof(int)));

  // Find allocated entries
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    find_alllocated_entries_CUDA(block_rect, thread_rect, dev_entries,
                                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH),
                                 this->dev_allocated_entries,
                                 this->dev_number_of_entries);
    //		CUDA_CKECK_KERNEL;

    // Find planar entries
  }

  // Generate 2D plane structure
  {
    //
    int total_plane_block_num = 0;
    //
    Plane_pixel *temp_plane_pixel_array = nullptr;
    checkCudaErrors(cudaMalloc(
        (void **)&temp_plane_pixel_array,
        PIXEL_BLOCK_NUM * PLANE_PIXEL_BLOCK_SIZE * sizeof(Plane_pixel)));

    checkCudaErrors(cudaMemcpy(&this->number_of_entries,
                               this->dev_number_of_entries, sizeof(int),
                               cudaMemcpyDeviceToHost));
    // printf("this->number_of_entries  = %d\n", this->number_of_entries);
    for (int plane_id = 1; plane_id < this->plane_counter; plane_id++) {
      checkCudaErrors(
          cudaMemset(this->dev_number_of_pixel_block, 0x00, sizeof(int)));
      checkCudaErrors(
          cudaMemset(this->dev_plane_entry_excess_counter, 0x00, sizeof(int)));
      checkCudaErrors(cudaMemset(
          temp_plane_pixel_array, 0x00,
          PIXEL_BLOCK_NUM * PLANE_PIXEL_BLOCK_SIZE * sizeof(Plane_pixel)));

      // Allocate memory
      PlaneHashEntry *temp_entry = nullptr;
      checkCudaErrors(
          cudaMalloc((void **)&temp_entry,
                     (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH) *
                         sizeof(PlaneHashEntry)));
      checkCudaErrors(
          cudaMemset(temp_entry, 0xFF,
                     (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH) *
                         sizeof(PlaneHashEntry)));
      this->dev_plane_entry_list.push_back(temp_entry);

      // PlaneHashEntry temp_buffer_1[ORDERED_PLANE_TABLE_LENGTH +
      // EXCESS_PLANE_TABLE_LENGTH]; checkCudaErrors(cudaMemcpy(temp_buffer_1,
      // temp_entry, 1024 * sizeof(PlaneHashEntry), cudaMemcpyDeviceToHost));
      // for (int i = 0; i < 1024; i++)
      //{
      //	printf("%d, %d, %d\n",
      //		   temp_buffer_1[i].position[0],
      //		   temp_buffer_1[i].position[1],
      //		   temp_buffer_1[i].ptr);
      //}

      // Allocate and re-sort entries
      int plane_block_num = 0, last_plane_block_num = 0, new_allocate_num = 1;
      while (new_allocate_num != 0) {
        // Prepare for allocation
        checkCudaErrors(cudaMemset(this->dev_pixel_block_need_allocate, 0x00,
                                   ORDERED_TABLE_LENGTH * sizeof(bool)));
        checkCudaErrors(
            cudaMemset(this->dev_pixel_block_buffer, 0x00,
                       ORDERED_TABLE_LENGTH * sizeof(My_Type::Vector2i)));

        // Build allocate pixel block list
        thread_rect.x = VOXEL_BLOCK_WDITH;
        thread_rect.y = VOXEL_BLOCK_WDITH;
        thread_rect.z = VOXEL_BLOCK_WDITH;
        block_rect.x = this->number_of_entries;
        block_rect.y = 1;
        block_rect.z = 1;
        build_allocate_flag_CUDA(
            block_rect, thread_rect, this->dev_allocated_entries,
            dev_voxel_array, temp_entry, this->plane_coordinate_list[plane_id],
            plane_id, this->dev_pixel_block_need_allocate,
            this->dev_pixel_block_buffer);
        //				CUDA_CKECK_KERNEL;

        // Allocate pixel blocks
        thread_rect.x = 256;
        thread_rect.y = 1;
        thread_rect.z = 1;
        block_rect.x = ORDERED_PLANE_TABLE_LENGTH / thread_rect.x;
        block_rect.y = 1;
        block_rect.z = 1;
        allocate_plane_blocks_CUDA(block_rect, thread_rect,
                                   this->dev_pixel_block_need_allocate,
                                   this->dev_pixel_block_buffer, temp_entry,
                                   this->dev_plane_entry_excess_counter,
                                   this->dev_number_of_pixel_block);
        //				CUDA_CKECK_KERNEL;

        //
        last_plane_block_num = plane_block_num;
        checkCudaErrors(cudaMemcpy(&plane_block_num,
                                   this->dev_number_of_pixel_block, sizeof(int),
                                   cudaMemcpyDeviceToHost));
        new_allocate_num = plane_block_num - last_plane_block_num;
      }
      // printf("plane_id : %d\tplane_block_num = %d\n", plane_id,
      // plane_block_num);
      total_plane_block_num += plane_block_num;

      //

      // Find allocated pixel block array
      checkCudaErrors(
          cudaMemset(this->dev_allocated_counter, 0x00, sizeof(int)));
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x = (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH) /
                     thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      find_allocated_planar_entries_CUDA(block_rect, thread_rect, temp_entry,
                                         this->dev_allocated_entry_index,
                                         this->dev_allocated_counter);
      //			CUDA_CKECK_KERNEL;
      // Fusion to plane hash array
      thread_rect.x = PLANE_PIXEL_BLOCK_WIDTH;
      thread_rect.y = PLANE_PIXEL_BLOCK_WIDTH;
      thread_rect.z = 1;
      block_rect.x = plane_block_num;
      block_rect.y = 1;
      block_rect.z = 1;
      fusion_sdf_to_plane_CUDA(block_rect, thread_rect, dev_entries,
                               dev_voxel_array, this->dev_allocated_entry_index,
                               this->plane_list[plane_id],
                               this->plane_coordinate_list[plane_id],
                               temp_entry, temp_plane_pixel_array);
      // CUDA_CKECK_KERNEL;

      // Compress plane hash array
      int number_of_pixel_block;
      checkCudaErrors(cudaMemcpy(&number_of_pixel_block,
                                 this->dev_number_of_pixel_block, sizeof(int),
                                 cudaMemcpyDeviceToHost));
      this->plane_array_length_list.push_back(number_of_pixel_block);
      //
      Plane_pixel *buffer_plane_pixel_array = nullptr;
      checkCudaErrors(
          cudaMalloc((void **)&buffer_plane_pixel_array,
                     number_of_pixel_block * PLANE_PIXEL_BLOCK_SIZE *
                         sizeof(Plane_pixel)));
      checkCudaErrors(cudaMemcpy(
          buffer_plane_pixel_array, temp_plane_pixel_array,
          number_of_pixel_block * PLANE_PIXEL_BLOCK_SIZE * sizeof(Plane_pixel),
          cudaMemcpyDeviceToDevice));
      this->dev_plane_pixel_array_list.push_back(buffer_plane_pixel_array);
    }
    //
    checkCudaErrors(cudaFree(temp_plane_pixel_array));
    // printf("total_plane_block_num = %d\n", total_plane_block_num);
  }
}

void Plane_map::generate_planar_block_render_information() {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Compute position of pixel blocks
  int number_of_blocks = 0;
  for (int plane_id = 1; plane_id < this->plane_array_length_list.size();
       plane_id++)
    number_of_blocks += this->plane_array_length_list[plane_id];

  // Allocate memory for block vertex array
  this->block_vertexs = (My_Type::Vector3f *)malloc(number_of_blocks * 8 *
                                                    sizeof(My_Type::Vector3f));
  checkCudaErrors(cudaMalloc((void **)&this->dev_block_vertexs,
                             number_of_blocks * 8 * sizeof(My_Type::Vector3f)));

  //
  checkCudaErrors(
      cudaMemset(this->dev_number_of_pixel_block, 0x00, sizeof(int)));
  // generate
  for (int plane_id = 1; plane_id < this->plane_counter; plane_id++) {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH) /
                   thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_block_vertex_CUDA(
        block_rect, thread_rect, this->dev_plane_entry_list[plane_id],
        this->plane_coordinate_list[plane_id], this->plane_list[plane_id],
        this->dev_block_vertexs, this->dev_number_of_pixel_block);
    // CUDA_CKECK_KERNEL;
  }

  // Copy out vertex array
  checkCudaErrors(cudaMemcpy(this->block_vertexs, this->dev_block_vertexs,
                             number_of_blocks * 8 * sizeof(My_Type::Vector3f),
                             cudaMemcpyDeviceToHost));

  // printf("number_of_planar_2D blocks = %d\n", number_of_blocks);
  this->number_of_pixel_blocks = number_of_blocks;
}
