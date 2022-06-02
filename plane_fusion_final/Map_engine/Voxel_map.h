/**
 *  @file Voxel_map.h
 *  @brief Voxel map
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include <Eigen/Dense>

#include "Map_engine/voxel_definition.h"
#include "OurLib/My_matrix.h"
#include "OurLib/My_pose.h"

/**
 * @brief The RaycastMode enum.
 */
enum RaycastMode {
  /** Raycast for tracking. */
  RAYCSAT_FOR_TRACKING = 0,
  /** Raycast for view. */
  RAYCAST_FOR_VIEW = 1,
};

/**
 * @brief The Voxel_map class, reference infinitam implementation.
 */
class Voxel_map {
 public:
  /** @brief Default constructor. */
  Voxel_map();
  /** @brief Default destructor. */
  ~Voxel_map();

  /** @brief Camera pose in this submap coordiante. */
  My_pose camera_pose_in_submap;

  /** @brief Flag whether current voxel block occupancy exceeds the total
   * amount. */
  bool out_of_block;
  /** @brief Upper limit number of voxel block. */
  int max_voxel_block_number;

  /** @brief Voxel block hash code collision number. */
  int collision_counter;
  /** @brief Voxel block hash code collision number. */
  int *dev_collision_counter;

  /** @brief CUDA device pointer of order entries and excess entries array. */
  HashEntry *dev_entrise;

  /** @brief Visible Hash entry list (block position) for view. */
  HashEntry *visible_list;
  HashEntry *dev_visible_list;

  /** @brief Device pointer of voxel block array.(VBA) */
  Voxel_f *dev_voxel_block_array;
  /** @brief Device pointer of relative voxel block list. */
  HashEntry *dev_relative_list;

  /** @brief Number of visible voxel blocks. */
  int number_of_visible_blocks;
  int *dev_number_of_visible_blocks;

  /** @brief Number of voxel blocks in current map. (Equal to next empty block
   * index). */
  int number_of_blocks;
  int *dev_number_of_blocks;

  /** @brief Number of relative blocks */
  int number_of_relative_blocks;
  int *dev_number_of_relative_blocks;

  /** @brief CUDA device pointer. The flag whether voxel block need to be
   * allocate. */
  char *dev_allocate_flag;
  /** @brief CUDA device pointer. The flag whether voxel block is visible. */
  char *dev_visible_flag;

  /** @brief CUDA device pointer. Ordered voxel block position. */
  My_Type::Vector3i *dev_ordered_position;
  /** @brief Voxel block position corresponds to voxel block. */
  My_Type::Vector3i *voxel_block_position;
  /** @brief CUDA device pointer of voxel_block_position. */
  My_Type::Vector3i *dev_voxel_block_position;

  int min_depth, max_depth, *dev_min_depth, *dev_max_depth;
  /** @brief Weight center of current depth frame's voxel block. */
  My_Type::Vector3f current_weight_center, *dev_current_weight_center;
  /** @brief Weight center of Map's voxel block. */
  My_Type::Vector3f map_weight_center, *dev_map_weight_center;

#if __unix__
#pragma region "Output, gengerated by raycast module" {
#elif _WIN32
#pragma region(Output, gengerated by raycast module)
#endif
  // --------------- For tracking ---------------
  /** @brief Width/Height  of range_map. */
  int raycast_range_map_width, raycast_range_map_height;

  /** @brief Range map for raycast full resolution depth image. */
  My_Type::Vector2f *raycast_range_map;
  My_Type::Vector2f *dev_raycast_range_map;

  int raycast_depth_width, raycast_depth_height;

  /** @brief Raycast points. */
  My_Type::Vector3f *raycast_points;
  My_Type::Vector3f *dev_raycast_points;
  /** @brief Raycast normal (caculate normal vector from SDF of voxel) */
  My_Type::Vector3f *raycast_normal;
  My_Type::Vector3f *dev_raycast_normal;
  /** @brief CUDA device pointer. Raycast points' fusiong weight. */
  int *dev_raycast_weight;
  /** @brief CUDA device pointer. Raycast points' plane label. */
  int *dev_raycast_plane_label;

  // --------------- For view ---------------

  /** @brief Width/Height of range_map. */
  int scene_range_map_width, scene_range_map_height;
  /** @brief Range map for raycast full resolution depth image. */
  My_Type::Vector2f *scene_range_map;
  My_Type::Vector2f *dev_scene_range_map;

  int scene_depth_width, scene_depth_height;
  /** @brief Raycast scene points. */
  My_Type::Vector3f *scene_points;
  My_Type::Vector3f *dev_scene_points;
  /** @brief CUDA device pointer. Raycast normal (caculate normal vector from
   * SDF of voxel). */
  My_Type::Vector3f *dev_scene_normals;
  /** @brief CUDA device pointer. Raycast points' fusion weight. */
  int *dev_scene_weight;
  /** @brief CUDA device pointer. Raycast points' plane label. */
  int *dev_scene_plane_label;

  ////! The color buffer array for (OpenGL) rendering scene_points.
  // My_Type::Vector4uc * scene_color;
  // My_Type::Vector4uc * dev_scene_color;

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Debug" {
#elif _WIN32
#pragma region(Debug)
#endif

  // for Debug
  char *allocate_flag;
  My_Type::Vector3i *ordered_position;
  int allocate_flag_counter;

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  /**
   * @brief Initialize voxel map object.
   * @param aligned_depth_size Size of depth image for tracking.
   * @param max_voxel_block_number
   */
  void init_Voxel_map(My_Type::Vector2i aligned_depth_size,
                      int max_voxel_block_number);

  /**
   * @brief Allocate voxel block by current frame points.
   * @param dev_current_points CUDA device pointer, current depth frame points
   *        cloud.
   * @param camera_pose The camera pose matrix in map coordinate.
   * @param submap_pose The submap pose matrix in world coordinate.
   * @return Number of voxel block allocated.
   */
  int allocate_voxel_block(
      My_Type::Vector3f *dev_current_points, Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f submap_pose = Eigen::Matrix4f::Identity());

  /**
   * @brief Fusion SDF value to voxel with known camera pose and current points.
   * @param dev_current_points CUDA device pointer. Current depth frame points
   *        cloud.
   * @param dev_current_normal CUDA device pointer. Current depth frame points
   *        cloud normal vectors.
   * @param camera_pose The camera pose matrix in map coordinate.
   * @param sumap_pose
   */
  void fusion_SDF_to_voxel(
      My_Type::Vector3f *dev_current_points,
      My_Type::Vector3f *dev_current_normal, Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f sumap_pose = Eigen::Matrix4f::Identity());

  /**
   * @brief Fusion plane label.
   * @param dev_current_points CUDA device pointer. Current depth frame points
            cloud.
   * @param plane_img CUDA device pointer. Current plane segmentation result.(A
   label mask image)
   * @param camera_pose The camera pose matrix in map coordinate.
   * @param sumap_pose
   */
  void fusion_plane_label_to_voxel(
      My_Type::Vector3f *dev_current_points, int *plane_img,
      Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f sumap_pose = Eigen::Matrix4f::Identity());

  /**
   * @brief Merge fragment voxel map to this voxel map.
   * @param fragment_map The fragment map object.
   * @param dev_plane_global_index
   */
  void merge_with_voxel_map(const Voxel_map &fragment_map,
                            int *dev_plane_global_index);

  /**
   * @brief Update from last voxel map (only update visible blocks)
   * @param camera_pose
   * @param dev_current_points
   * @param dev_entries
   * @param dev_voxel_array
   */
  void update_from_last_voxel_map(My_pose &camera_pose,
                                  My_Type::Vector3f *dev_current_points,
                                  HashEntry *dev_entries,
                                  Voxel_f *dev_voxel_array);

  /**
   * @brief Raycast all information in voxel to several image matrices.
   * @note Raycast is the inverse operation of projection. See details on
           wiki (https://en.wikipedia.org/wiki/Ray_casting).
   * @param camera_pose The camera pose matrix in map coordinate.
   * @param mode Raycast mode. See Raycast_flag.
   */
  void raycast_by_pose(Eigen::Matrix4f camera_pose, RaycastMode mode);

  /**
   * @brief Set map to empty statement.
   */
  void clear_Voxel_map();

  /**
   * @brief Compress map.
   */
  void compress_voxel_map();

  /**
   * @brief Release voxel map
   */
  void release_voxel_map();

  ////! Raycast informations for (OpenGL) rendering Voxel map and caculate
  /// points color array.
  ///*!
  //	\param	pose				The camera pose matrix in map
  // coordinate.

  //	\param	render_mode			Mode of render. See details.

  //	\return	void
  //*/
  // void render_map(Eigen::Matrix4f trans_mat, int render_mode);

  // void render_map_special(Eigen::Matrix4f trans_mat, int render_mode, int
  // fragment_map_id);

  /**
   * @brief Copy out data for debug or showing result.
   */
  void extract_visible_list_to_CPU();

  /**
   * @brief Copy out data for debug or showing result.
   */
  void copy_out_raycast_points();

  /**
   * @brief Reduce VoxelBlock's weight center.
   */
  void reduce_Voxel_map_weight_center();
};
