#pragma once

//
#include "Map_engine/Mesh_generator.h"
#include "Map_engine/voxel_definition.h"
#include "Plane_detector/Plane_structure.h"
#include "SLAM_system/SLAM_system_settings.h"
//
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include "OurLib/My_matrix.h"
//
#include <vector>

//!
/*!


*/
class Mesh_generator {
 public:
#pragma region(Non - planar region)
  //! Triangle vertex
  My_Type::Vector3f *triangles, *dev_triangles;
  //!	Normal vector of triangle vertex
  My_Type::Vector3f *triangle_normals, *dev_triangle_normals;
  //! Pseudo color array of triangle vertex
  My_Type::Vector4uc *triangle_color, *dev_triangle_color;

  //! Number of triangles
  int number_of_triangles, *dev_number_of_triangles;

  //!
  HashEntry *dev_allocated_entries, *dev_nonplanar_entries;
  //! Number of used entries
  int number_of_entries, *dev_number_of_entries;
  //!
  int number_of_nonplanar_blocks, *dev_number_of_nonplanar_blocks;

  //
  const int max_number_of_triangle =
      (int)SUBMAP_VOXEL_BLOCK_NUM * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH * 4;
#pragma endregion

#pragma region(Planar region)
  //! Planar triangle vertex
  My_Type::Vector3f *planar_triangles, *dev_planar_triangles;
  //!	Normal vector ofplanar triangle vertex
  My_Type::Vector3f *planar_triangle_normals, *dev_planar_triangle_normals;
  //! Pseudo color array of planar triangle vertex
  My_Type::Vector4uc *planar_triangle_color, *dev_planar_triangle_color;

  //! Number of planar triangles
  int number_of_planar_triangles, *dev_number_of_planar_triangles;
  //!
  std::vector<int> planar_triangle_offset_list;

#pragma endregion

  //
  Mesh_generator();
  ~Mesh_generator();

  //! Generate triangle mesh from TSDF
  /*!

  */
  void generate_mesh_from_voxel(
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
      std::vector<My_Type::Vector2i> relabel_plane_list =
          std::vector<My_Type::Vector2i>());

  //! Generate Non-planar triangle mesh from TSDF
  /*!

  */
  void generate_nonplanar_mesh_from_voxel(HashEntry *dev_entry,
                                          Voxel_f *dev_voxel_block_array);

  //! Generate triangle mesh from PlaneHash structure
  /*!

  */
  void generate_mesh_from_plane_array(
      std::vector<Plane_info> &plane_list,
      std::vector<Plane_coordinate> &plane_coordiante,
      std::vector<PlaneHashEntry *> &dev_plane_entry_list,
      std::vector<Plane_pixel *> &dev_plane_pixel_array_list,
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array);

  //!
  /*!

  */
  void copy_triangle_mesh_from(Mesh_generator *src_mesh, int this_offset);
  void copy_triangle_mesh_from(
      Mesh_generator *src_mesh, int this_offset, Eigen::Matrix4f submap_pose,
      HashEntry *dev_entry, Voxel_f *dev_voxel_block_array,
      std::vector<My_Type::Vector2i> relabel_plane_list);

  //!
  /*!

  */
  void compress_mesh();
};
