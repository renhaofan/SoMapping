#include "Mesh_generator.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "Map_engine/Mesh_structure.h"
#include "Map_engine/voxel_definition.h"
#include "OurLib/My_matrix.h"
#include "OurLib/my_math_functions.h"

//
#include "Map_engine/Mesh_generator_KernelFunc.cuh"

//
Mesh_generator::Mesh_generator() {
  this->number_of_triangles = 0;
  this->number_of_planar_triangles = 0;
  this->planar_triangle_offset_list.clear();
  this->planar_triangle_offset_list.push_back(0);

  // Host memory
  //
  this->triangles = (My_Type::Vector3f *)malloc(this->max_number_of_triangle *
                                                3 * sizeof(My_Type::Vector3f));
  this->triangle_normals = (My_Type::Vector3f *)malloc(
      this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f));
  this->triangle_color = (My_Type::Vector4uc *)malloc(
      this->max_number_of_triangle * 3 * sizeof(My_Type::Vector4uc));
  //
  this->planar_triangles = (My_Type::Vector3f *)malloc(
      this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f));
  this->planar_triangle_normals = (My_Type::Vector3f *)malloc(
      this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f));
  this->planar_triangle_color = (My_Type::Vector4uc *)malloc(
      this->max_number_of_triangle * 3 * sizeof(My_Type::Vector4uc));

  // Device memory
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_triangles),
                 this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_triangle_normals),
                 this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_triangle_color),
                             this->max_number_of_triangle * 3 *
                                 sizeof(My_Type::Vector4uc)));
  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_triangles), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_allocated_entries),
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_nonplanar_entries),
                             (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) *
                                 sizeof(HashEntry)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_number_of_entries), sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_number_of_nonplanar_blocks),
                             sizeof(int)));
  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_planar_triangles),
                 this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_planar_triangle_normals),
                 this->max_number_of_triangle * 3 * sizeof(My_Type::Vector3f)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_planar_triangle_color),
                             this->max_number_of_triangle * 3 *
                                 sizeof(My_Type::Vector4uc)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_number_of_planar_triangles),
                             sizeof(int)));

  // printf("--- Memory cost = %d + %d = %d MB\n",
  //	   this->max_number_of_triangle * 3 * (sizeof(My_Type::Vector3f) +
  //sizeof(My_Type::Vector3f) + sizeof(My_Type::Vector4uc)) >> 20,
  //	   (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(HashEntry) >>
  //20, 	   (this->max_number_of_triangle * 3 * (sizeof(My_Type::Vector3f) +
  //sizeof(My_Type::Vector3f) + sizeof(My_Type::Vector4uc)) +
  //	   (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) * sizeof(HashEntry)) >>
  //20);
}
Mesh_generator::~Mesh_generator() {
  //
  free(this->triangles);
  free(this->triangle_normals);
  free(this->triangle_color);
  free(this->planar_triangles);
  free(this->planar_triangle_normals);
  free(this->planar_triangle_color);

  //
  checkCudaErrors(cudaFree(this->dev_triangles));
  checkCudaErrors(cudaFree(this->dev_triangle_normals));
  checkCudaErrors(cudaFree(this->dev_triangle_color));
  checkCudaErrors(cudaFree(this->dev_number_of_triangles));
  checkCudaErrors(cudaFree(this->dev_allocated_entries));
  checkCudaErrors(cudaFree(this->dev_nonplanar_entries));
  checkCudaErrors(cudaFree(this->dev_number_of_entries));
  checkCudaErrors(cudaFree(this->dev_number_of_nonplanar_blocks));
  //
  checkCudaErrors(cudaFree(this->dev_planar_triangles));
  checkCudaErrors(cudaFree(this->dev_planar_triangle_normals));
  checkCudaErrors(cudaFree(this->dev_planar_triangle_color));
  checkCudaErrors(cudaFree(this->dev_number_of_planar_triangles));
}

//
void Mesh_generator::generate_mesh_from_voxel(
    HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
    std::vector<My_Type::Vector2i> relabel_plane_list) {
  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(cudaMemset(this->dev_number_of_triangles, 0x00, sizeof(int)));
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
    find_alllocated_entries_CUDA(block_rect, thread_rect, dev_entry,
                                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH),
                                 this->dev_allocated_entries,
                                 this->dev_number_of_entries);
    //		CUDA_CKECK_KERNEL;
  }

  // Marching cube process
  {
    checkCudaErrors(cudaMemcpy(&this->number_of_entries,
                               this->dev_number_of_entries, sizeof(int),
                               cudaMemcpyDeviceToHost));
    // printf("this->number_of_entries  = %d\n", this->number_of_entries);

    thread_rect.x = VOXEL_BLOCK_WDITH;
    thread_rect.y = VOXEL_BLOCK_WDITH;
    thread_rect.z = VOXEL_BLOCK_WDITH;
    block_rect.x = this->number_of_entries;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_triangle_mesh_CUDA(
        block_rect, thread_rect, dev_entry, this->dev_allocated_entries,
        dev_voxel_block_array, this->dev_triangles,
        this->dev_number_of_triangles, this->max_number_of_triangle);
    //		CUDA_CKECK_KERNEL;
  }

  // Generate normal vectors of each vertex
  {
    checkCudaErrors(cudaMemcpy(&this->number_of_triangles,
                               this->dev_number_of_triangles, sizeof(int),
                               cudaMemcpyDeviceToHost));
    // printf("number_of_triangles  = %d\n", this->number_of_triangles);

    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(this->number_of_triangles * 3, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_normals_CUDA(block_rect, thread_rect, this->dev_triangles,
                                 dev_entry, dev_voxel_block_array,
                                 this->dev_triangle_normals);
    //		CUDA_CKECK_KERNEL;
  }

  // Generate plane label pseudo color array
  if (relabel_plane_list.size() > 0) {
    My_Type::Vector2i *dev_relabel_plane_list;
    checkCudaErrors(
        cudaMalloc((void **)&(dev_relabel_plane_list),
                   relabel_plane_list.size() * sizeof(My_Type::Vector2i)));
    checkCudaErrors(
        cudaMemcpy(dev_relabel_plane_list, relabel_plane_list.data(),
                   relabel_plane_list.size() * sizeof(My_Type::Vector2i),
                   cudaMemcpyHostToDevice));

    //
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(this->number_of_triangles * 3, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_color_CUDA(block_rect, thread_rect, this->dev_triangles,
                               this->number_of_triangles * 3, dev_entry,
                               dev_voxel_block_array, dev_relabel_plane_list,
                               this->dev_triangle_color);
    //		CUDA_CKECK_KERNEL;

    checkCudaErrors(cudaFree(dev_relabel_plane_list));
  } else {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(this->number_of_triangles * 3, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_color_CUDA(block_rect, thread_rect, this->dev_triangles,
                               this->number_of_triangles * 3, dev_entry,
                               dev_voxel_block_array, this->dev_triangle_color);
    //		CUDA_CKECK_KERNEL;
  }

  // Copy out data
  {
    checkCudaErrors(
        cudaMemcpy(this->triangles, this->dev_triangles,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                   cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(this->triangle_normals, this->dev_triangle_normals,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                   cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(this->triangle_color, this->dev_triangle_color,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector4uc),
                   cudaMemcpyDeviceToHost));
  }
}

//
void Mesh_generator::generate_nonplanar_mesh_from_voxel(
    HashEntry *dev_entry, Voxel_f *dev_voxel_block_array) {
  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  checkCudaErrors(cudaMemset(this->dev_number_of_triangles, 0x00, sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_number_of_entries, 0x00, sizeof(int)));
  checkCudaErrors(
      cudaMemset(this->dev_number_of_nonplanar_blocks, 0x00, sizeof(int)));

  // Find allocated entries
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH) / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    find_alllocated_entries_CUDA(block_rect, thread_rect, dev_entry,
                                 (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH),
                                 this->dev_allocated_entries,
                                 this->dev_number_of_entries);
    //		CUDA_CKECK_KERNEL;
  }

  // Find non-planar entries
  {
    checkCudaErrors(cudaMemcpy(&this->number_of_entries,
                               this->dev_number_of_entries, sizeof(int),
                               cudaMemcpyDeviceToHost));
    //
    thread_rect.x = VOXEL_BLOCK_WDITH;
    thread_rect.y = VOXEL_BLOCK_WDITH;
    thread_rect.z = VOXEL_BLOCK_WDITH;
    block_rect.x = this->number_of_entries;
    block_rect.y = 1;
    block_rect.z = 1;
    //
    find_nonplanar_blocks_CUDA(
        block_rect, thread_rect, dev_entry, this->dev_allocated_entries,
        dev_voxel_block_array, this->dev_nonplanar_entries,
        this->dev_number_of_nonplanar_blocks);
    //		CUDA_CKECK_KERNEL;

    //
    checkCudaErrors(cudaMemcpy(&this->number_of_nonplanar_blocks,
                               this->dev_number_of_nonplanar_blocks,
                               sizeof(int), cudaMemcpyDeviceToHost));
  }
  printf("non-planar blocks = %d\n", this->number_of_nonplanar_blocks);
}

//
void Mesh_generator::generate_mesh_from_plane_array(
    std::vector<Plane_info> &plane_list,
    std::vector<Plane_coordinate> &plane_coordiante,
    std::vector<PlaneHashEntry *> &dev_plane_entry_list,
    std::vector<Plane_pixel *> &dev_plane_pixel_array_list,
    HashEntry *dev_entry, Voxel_f *dev_voxel_block_array) {
  //
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  //
  this->planar_triangle_offset_list.clear();
  this->planar_triangle_offset_list.push_back(0);
  // Generate mesh for each plane
  for (int plane_id = 1; plane_id < dev_plane_entry_list.size(); plane_id++) {
    int number_of_triagnles_bffer = this->planar_triangle_offset_list.back();
    checkCudaErrors(cudaMemcpy(this->dev_number_of_planar_triangles,
                               &number_of_triagnles_bffer, sizeof(int),
                               cudaMemcpyHostToDevice));

    //
    thread_rect.x = PLANE_PIXEL_BLOCK_WIDTH;
    thread_rect.y = PLANE_PIXEL_BLOCK_WIDTH;
    thread_rect.z = 1;
    block_rect.x = ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_triangle_mesh_from_plane_CUDA(
        block_rect, thread_rect, plane_list[plane_id],
        plane_coordiante[plane_id], dev_plane_entry_list[plane_id],
        dev_plane_pixel_array_list[plane_id], this->dev_planar_triangles,
        this->dev_number_of_planar_triangles);
    //		CUDA_CKECK_KERNEL;

    //
    int triangle_offset;
    checkCudaErrors(cudaMemcpy(&triangle_offset,
                               this->dev_number_of_planar_triangles,
                               sizeof(int), cudaMemcpyDeviceToHost));
    this->planar_triangle_offset_list.push_back(triangle_offset);

    // printf("triangle_offset = %d\n", triangle_offset);
  }
  this->number_of_planar_triangles = this->planar_triangle_offset_list.back();

  // Generate normal vectors of each vertex
  {
    checkCudaErrors(cudaMemcpy(&this->number_of_planar_triangles,
                               this->dev_number_of_planar_triangles,
                               sizeof(int), cudaMemcpyDeviceToHost));
    // printf("number_of_triangles  = %d\n", this->number_of_planar_triangles);

    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(this->number_of_planar_triangles * 3, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_normals_CUDA(
        block_rect, thread_rect, this->dev_planar_triangles, dev_entry,
        dev_voxel_block_array, this->dev_planar_triangle_normals);
    //		CUDA_CKECK_KERNEL;
  }

  // Copy out data
  {
    checkCudaErrors(cudaMemcpy(
        this->planar_triangles, this->dev_planar_triangles,
        this->number_of_planar_triangles * 3 * sizeof(My_Type::Vector3f),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(
        this->planar_triangle_normals, this->dev_planar_triangle_normals,
        this->number_of_planar_triangles * 3 * sizeof(My_Type::Vector3f),
        cudaMemcpyDeviceToHost));
  }
}

//
void Mesh_generator::compress_mesh() {
  //! Non-planar region
  {
    // printf("Triangle mesh re-allocate rate : %f\n",
    // (float)this->number_of_triangles / (float)this->max_number_of_triangle);
    // printf("Triangle mesh re-allocate : %d / %d \n",
    // this->number_of_triangles, this->max_number_of_triangle);
    printf("Triangle mesh size : %.3f\t MB\n",
           (float)(this->number_of_triangles * sizeof(My_Type::Vector3f) * 6) /
               (1024.0f * 1024.0f));
    printf("Voxel array size :	 %.3f\t MB\n",
           (float)(this->number_of_entries * (4 + 4) * (8 * 8 * 8)) /
               (1024.0f * 1024.0f));
    printf("Voxel block number : %d\t blocks\n", this->number_of_entries);

    My_Type::Vector3f *temp_ptr = nullptr;
    // Re-allocate device memory
    // Triangle mesh vertex position array
    checkCudaErrors(
        cudaMalloc((void **)&(temp_ptr),
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f)));
    checkCudaErrors(
        cudaMemcpy(temp_ptr, this->dev_triangles,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                   cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(this->dev_triangles));
    this->dev_triangles = temp_ptr;
    // Triangle mesh vertex normals array
    checkCudaErrors(
        cudaMalloc((void **)&(temp_ptr),
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f)));
    checkCudaErrors(
        cudaMemcpy(temp_ptr, this->dev_triangle_normals,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                   cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(this->dev_triangle_normals));
    this->dev_triangle_normals = temp_ptr;
    // Triagle vertex color array
    My_Type::Vector4uc *temp_color_ptr;
    checkCudaErrors(
        cudaMalloc((void **)&(temp_color_ptr),
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector4uc)));
    checkCudaErrors(
        cudaMemcpy(temp_color_ptr, this->dev_triangle_color,
                   this->number_of_triangles * 3 * sizeof(My_Type::Vector4uc),
                   cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(this->dev_triangle_color));
    this->dev_triangle_color = temp_color_ptr;

    // Re-allocate host memory
    // Triangle mesh vertex position array
    temp_ptr = (My_Type::Vector3f *)malloc(this->number_of_triangles * 3 *
                                           sizeof(My_Type::Vector3f));
    memcpy(temp_ptr, this->triangles,
           this->number_of_triangles * 3 * sizeof(My_Type::Vector3f));
    free(this->triangles);
    this->triangles = temp_ptr;
    // Triangle mesh vertex normals array
    temp_ptr = (My_Type::Vector3f *)malloc(this->number_of_triangles * 3 *
                                           sizeof(My_Type::Vector3f));
    memcpy(temp_ptr, this->triangle_normals,
           this->number_of_triangles * 3 * sizeof(My_Type::Vector3f));
    free(this->triangle_normals);
    this->triangle_normals = temp_ptr;
    // Triangle mesh vertex color array
    temp_color_ptr = (My_Type::Vector4uc *)malloc(
        this->number_of_triangles * 3 * sizeof(My_Type::Vector4uc));
    memcpy(temp_color_ptr, this->triangle_color,
           this->number_of_triangles * 3 * sizeof(My_Type::Vector4uc));
    free(this->triangle_color);
    this->triangle_color = temp_color_ptr;
  }

  //! Planar region
  if (false) {
    My_Type::Vector3f *temp_ptr = nullptr;
    // Re-allocate device memory
    // Triangle mesh vertex position array
    checkCudaErrors(
        cudaMalloc((void **)&(temp_ptr), this->number_of_planar_triangles * 3 *
                                             sizeof(My_Type::Vector3f)));
    checkCudaErrors(cudaMemcpy(temp_ptr, this->dev_planar_triangles,
                               this->number_of_planar_triangles * 3 *
                                   sizeof(My_Type::Vector3f),
                               cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(this->dev_triangles));
    this->dev_planar_triangles = temp_ptr;

    // Triangle mesh vertex position array
    temp_ptr = (My_Type::Vector3f *)malloc(this->number_of_planar_triangles *
                                           3 * sizeof(My_Type::Vector3f));
    memcpy(temp_ptr, this->planar_triangles,
           this->number_of_planar_triangles * 3 * sizeof(My_Type::Vector3f));
    free(this->planar_triangles);
    this->planar_triangles = temp_ptr;
  }
}

//
void Mesh_generator::copy_triangle_mesh_from(Mesh_generator *src_mesh,
                                             int this_offset) {
  //
  My_Type::Vector3f *point_ptr = this->triangles;
  point_ptr += this_offset * 3;
  memcpy(point_ptr, src_mesh->triangles,
         src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector3f));
  //
  My_Type::Vector3f *normal_ptr = this->triangle_normals;
  normal_ptr += this_offset * 3;
  memcpy(normal_ptr, src_mesh->triangle_normals,
         src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector3f));
  //
  My_Type::Vector4uc *color_ptr = this->triangle_color;
  color_ptr += this_offset * 3;
  memcpy(color_ptr, src_mesh->triangle_color,
         src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector4uc));
  //
  this->number_of_triangles = this_offset + src_mesh->number_of_triangles;

  //
  point_ptr = this->planar_triangles;
  point_ptr += this_offset * 3;
  memcpy(point_ptr, src_mesh->planar_triangles,
         src_mesh->number_of_planar_triangles * 3 * sizeof(My_Type::Vector3f));
  //
  normal_ptr = this->planar_triangle_normals;
  normal_ptr += this_offset * 3;
  memcpy(normal_ptr, src_mesh->planar_triangle_normals,
         src_mesh->number_of_planar_triangles * 3 * sizeof(My_Type::Vector3f));
  //
  color_ptr = this->planar_triangle_color;
  color_ptr += this_offset * 3;
  memcpy(color_ptr, src_mesh->planar_triangle_color,
         src_mesh->number_of_planar_triangles * 3 * sizeof(My_Type::Vector4uc));
  //
}

//
void Mesh_generator::copy_triangle_mesh_from(
    Mesh_generator *src_mesh, int this_offset, Eigen::Matrix4f submap_pose,
    HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
    std::vector<My_Type::Vector2i> relabel_plane_list) {
  My_Type::Vector2i *dev_relabel_plane_list;
  checkCudaErrors(cudaMalloc((void **)&(dev_relabel_plane_list),
                             128 * sizeof(My_Type::Vector2i)));
  checkCudaErrors(cudaMemset(dev_relabel_plane_list, 0x00,
                             128 * sizeof(My_Type::Vector2i)));
  //
  checkCudaErrors(
      cudaMemcpy(dev_relabel_plane_list, relabel_plane_list.data(),
                 relabel_plane_list.size() * sizeof(My_Type::Vector2i),
                 cudaMemcpyHostToDevice));

  //
  dim3 block_rect, thread_rect;
  //
  My_Type::Matrix44f submap_pose_mat;
  for (int i = 0; i < 16; i++)
    submap_pose_mat.data[i] = submap_pose.data()[i];

  //
  My_Type::Vector3f *dev_point_ptr = this->dev_triangles;
  dev_point_ptr += this_offset * 3;
  My_Type::Vector3f *dev_normal_ptr = this->dev_triangle_normals;
  dev_normal_ptr += this_offset * 3;
  {
    //
    thread_rect = dim3(256, 1, 1);
    block_rect.x =
        ceil_by_stride(src_mesh->number_of_triangles * 3, thread_rect.x);
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch CUDA kernel function
    copy_mesh_to_global_map_CUDA(
        block_rect, thread_rect, src_mesh->dev_triangles,
        src_mesh->dev_triangle_normals, submap_pose_mat,
        src_mesh->number_of_triangles * 3, dev_point_ptr, dev_normal_ptr);
    // CUDA_CKECK_KERNEL;
  }
  //
  My_Type::Vector3f *point_ptr = this->triangles;
  point_ptr += this_offset * 3;
  My_Type::Vector3f *normal_ptr = this->triangle_normals;
  normal_ptr += this_offset * 3;
  //
  checkCudaErrors(
      cudaMemcpy(point_ptr, dev_point_ptr,
                 src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                 cudaMemcpyDeviceToHost));
  checkCudaErrors(
      cudaMemcpy(normal_ptr, dev_normal_ptr,
                 src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector3f),
                 cudaMemcpyDeviceToHost));
  //
  My_Type::Vector4uc *dev_color_ptr = this->dev_triangle_color;
  dev_color_ptr += this_offset * 3;
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        ceil_by_stride(src_mesh->number_of_triangles * 3, thread_rect.x) /
        thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_color_CUDA(block_rect, thread_rect, src_mesh->dev_triangles,
                               src_mesh->number_of_triangles * 3, dev_entry,
                               dev_voxel_block_array, dev_relabel_plane_list,
                               dev_color_ptr);
    // CUDA_CKECK_KERNEL;
  }
  My_Type::Vector4uc *color_ptr = this->triangle_color;
  color_ptr += this_offset * 3;
  checkCudaErrors(
      cudaMemcpy(color_ptr, dev_color_ptr,
                 src_mesh->number_of_triangles * 3 * sizeof(My_Type::Vector4uc),
                 cudaMemcpyDeviceToHost));
  //
  this->number_of_triangles = this_offset + src_mesh->number_of_triangles;
  printf("this->number_of_triangles  = %d\n", this->number_of_triangles);

  //
  dev_point_ptr = this->dev_planar_triangles;
  point_ptr += this_offset * 3;
  dev_normal_ptr = this->dev_planar_triangle_normals;
  normal_ptr += this_offset * 3;
  //
  thread_rect = dim3(256, 1, 1);
  block_rect.x =
      ceil_by_stride(src_mesh->number_of_planar_triangles * 3, thread_rect.x);
  block_rect.y = 1;
  block_rect.z = 1;
  // Lunch CUDA kernel function
  copy_mesh_to_global_map_CUDA(
      block_rect, thread_rect, src_mesh->dev_planar_triangles,
      src_mesh->dev_planar_triangle_normals, submap_pose_mat,
      src_mesh->number_of_planar_triangles * 3, dev_point_ptr, dev_normal_ptr);
  // CUDA_CKECK_KERNEL;
  //
  point_ptr = this->planar_triangles;
  point_ptr += this_offset * 3;
  normal_ptr = this->planar_triangle_normals;
  normal_ptr += this_offset * 3;
  //
  checkCudaErrors(cudaMemcpy(point_ptr, dev_point_ptr,
                             src_mesh->number_of_planar_triangles * 3 *
                                 sizeof(My_Type::Vector3f),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normal_ptr, dev_normal_ptr,
                             src_mesh->number_of_planar_triangles * 3 *
                                 sizeof(My_Type::Vector3f),
                             cudaMemcpyDeviceToHost));
  //
  //
  dev_color_ptr = this->dev_planar_triangle_color;
  dev_color_ptr += this_offset * 3;
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = ceil_by_stride(src_mesh->number_of_planar_triangles * 3,
                                  thread_rect.x) /
                   thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    generate_vertex_color_CUDA(
        block_rect, thread_rect, src_mesh->dev_planar_triangles,
        src_mesh->number_of_planar_triangles * 3, dev_entry,
        dev_voxel_block_array, dev_relabel_plane_list, dev_color_ptr);
    // CUDA_CKECK_KERNEL;
  }
  color_ptr = this->planar_triangle_color;
  color_ptr += this_offset * 3;
  checkCudaErrors(cudaMemcpy(color_ptr, dev_color_ptr,
                             src_mesh->number_of_planar_triangles * 3 *
                                 sizeof(My_Type::Vector4uc),
                             cudaMemcpyDeviceToHost));

  //
  checkCudaErrors(cudaFree(dev_relabel_plane_list));
}