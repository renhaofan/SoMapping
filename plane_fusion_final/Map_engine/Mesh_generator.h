/**
 *  @file Mesh_generator.h
 *  @brief Generate mesh from TSDF voxel.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include "Map_engine/Mesh_generator.h"
#include "Map_engine/voxel_definition.h"
#include "Plane_detector/Plane_structure.h"
#include "SLAM_system/SLAM_system_settings.h"

#include <vector>
#include "OurLib/My_matrix.h"

class Mesh_generator {
 public:
#if __unix__
#pragma region "Non - planar region" {
#elif _WIN32
#pragma region(Non - planar region)
#endif
  /** @brief Triangle vertex on CPU. */
  My_Type::Vector3f *triangles;
  /** @brief Triangle vertex on GPU. */
  My_Type::Vector3f *dev_triangles;

  /** @brief Normal vector of triangle vertex on CPU. */
  My_Type::Vector3f *triangle_normals;
  /** @brief Normal vector of triangle vertex on GPU. */
  My_Type::Vector3f *dev_triangle_normals;

  /** @brief Pseudo color array of triangle vertex on CPU. */
  My_Type::Vector4uc *triangle_color;
  /** @brief Pseudo color array of triangle vertex on GPU. */
  My_Type::Vector4uc *dev_triangle_color;

  /** @brief Number of triangles on CPU. */
  int number_of_triangles;
  /** @brief Number of triangles on GPU. */
  int *dev_number_of_triangles;

  HashEntry *dev_allocated_entries, *dev_nonplanar_entries;

  /** @brief Number of used entries on CPU. */
  int number_of_entries;
  /** @brief Number of used entries on GPU. */
  int *dev_number_of_entries;

  int number_of_nonplanar_blocks, *dev_number_of_nonplanar_blocks;

  const int max_number_of_triangle =
      (int)SUBMAP_VOXEL_BLOCK_NUM * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH * 4;

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Planar region" {
#elif _WIN32
#pragma region(Planar region)
#endif
  /** @brief Planar triangle vertex. */
  My_Type::Vector3f *planar_triangles, *dev_planar_triangles;
  /** @brief Normal vector of planar triangle vertex. */
  My_Type::Vector3f *planar_triangle_normals, *dev_planar_triangle_normals;
  /** @brief Pseudo color array of planar triangle vertex. */
  My_Type::Vector4uc *planar_triangle_color, *dev_planar_triangle_color;

  /** @brief Number of planar triangles. */
  int number_of_planar_triangles, *dev_number_of_planar_triangles;
  std::vector<int> planar_triangle_offset_list;

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  Mesh_generator();
  ~Mesh_generator();

  /** @brief Generate triangle mesh from TSDF. */
  void generate_mesh_from_voxel(
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
      std::vector<My_Type::Vector2i> relabel_plane_list =
          std::vector<My_Type::Vector2i>());

  /** @brief Generate Non-planar triangle mesh from TSDF. */
  void generate_nonplanar_mesh_from_voxel(HashEntry *dev_entry,
                                          Voxel_f *dev_voxel_block_array);

  /** @brief Generate triangle mesh from PlaneHash structure. */
  void generate_mesh_from_plane_array(
      std::vector<Plane_info> &plane_list,
      std::vector<Plane_coordinate> &plane_coordiante,
      std::vector<PlaneHashEntry *> &dev_plane_entry_list,
      std::vector<Plane_pixel *> &dev_plane_pixel_array_list,
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array);

  void copy_triangle_mesh_from(Mesh_generator *src_mesh, int this_offset);
  void copy_triangle_mesh_from(
      Mesh_generator *src_mesh, int this_offset, Eigen::Matrix4f submap_pose,
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
      std::vector<My_Type::Vector2i> relabel_plane_list);

  void compress_mesh();
};
