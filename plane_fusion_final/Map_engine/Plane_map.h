/**
 *  @file Plane_map.h
 *  @brief Plane map
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include "Map_engine/voxel_definition.h"
#include "OurLib/My_matrix.h"
#include "Plane_detector/Plane_structure.h"

#include <iostream>
#include <vector>

/**
 * @brief The Plane_map class
 */
class Plane_map {
 public:
  /**
   * @brief Plane counter.
   * @details Label-0 is reserved for non-plane region.
   */
  int plane_counter = 1;

  /** @brief Plane list. */
  std::vector<Plane_info> plane_list;
  Plane_info *dev_plane_list;

  HashEntry *dev_allocated_entries;
  /** @brief Number of used entries. */
  int number_of_entries, *dev_number_of_entries;

  /** @brief Plane coordinate. */
  std::vector<Plane_coordinate> plane_coordinate_list;

  std::vector<PlaneHashEntry *> dev_plane_entry_list;
  std::vector<My_Type::Vector3f *> plane_block_position_list;
  int *dev_plane_entry_excess_counter;

  std::vector<int> plane_array_length_list;
  /** @brief HashPlane arrays. */
  std::vector<Plane_pixel *> dev_plane_pixel_array_list;

  int *dev_number_of_pixel_block;

  bool *dev_pixel_block_need_allocate;
  My_Type::Vector2i *dev_pixel_block_buffer;

  int *dev_allocated_entry_index, *dev_allocated_counter;

  int number_of_pixel_blocks;
  My_Type::Vector3f *block_vertexs, *dev_block_vertexs;

  Plane_map();
  ~Plane_map();

  void init();

  void update_plane_list(const Plane_info *current_planes,
                         std::vector<My_Type::Vector2i> &matches);

  void generate_plane_map(const HashEntry *dev_entries,
                          const Voxel_f *dev_voxel_array);

  void generate_planar_block_render_information();

 private:
  // void compress_hash_plane();
};
