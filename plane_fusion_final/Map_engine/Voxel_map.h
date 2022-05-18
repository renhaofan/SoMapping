
#pragma once

// Eigen
#include <Eigen/Dense>

//
#include "Map_engine/voxel_definition.h"
//
#include "OurLib/My_matrix.h"
#include "OurLib/My_pose.h"

//!
/*!

*/
enum RaycastMode {
  RAYCSAT_FOR_TRACKING = 0,
  RAYCAST_FOR_VIEW = 1,
};

//! A class for voxel map
/*!
        \brief		This class defines voxel map.

        \details	\image html 5_Hash_Table.svg
                                \n

*/
class Voxel_map {
 public:
  /*! Default constructor.	*/
  Voxel_map();
  /*! Default destructor.	*/
  ~Voxel_map();

  //! Pose of map
  // My_pose submap_pose;
  //! Camera pose in this submap coordiante
  My_pose camera_pose_in_submap;

  //! Marks if current voxel block occupancy exceeds the total amount.
  bool out_of_block;
  //!
  int max_voxel_block_number;

  //! Voxel block hash code collision number.
  int collision_counter;
  int *dev_collision_counter;

  //! CUDA device pointer of order entries and excess entries array.
  HashEntry *dev_entrise;

  //! Visible Hash entry list (block position) for view.
  HashEntry *visible_list;
  HashEntry *dev_visible_list;

  //! Device pointer of voxel block array.(VBA)
  Voxel_f *dev_voxel_block_array;
  //! Device pointer of relative voxel block list
  HashEntry *dev_relative_list;

  //! Number of visible voxel blocks.
  int number_of_visible_blocks;
  int *dev_number_of_visible_blocks;
  //! Number of voxel blocks in current map. (Equal to next empty block index)
  int number_of_blocks;
  int *dev_number_of_blocks;
  //! Number of relative blocks
  int number_of_relative_blocks;
  int *dev_number_of_relative_blocks;

  //! CUDA device pointer. The flag whether voxel block need to be allocate.
  char *dev_allocate_flag;
  //! CUDA device pointer. The flag whether voxel block is visible.
  char *dev_visible_flag;
  //! CUDA device pointer. Ordered voxel block position.
  My_Type::Vector3i *dev_ordered_position;
  //! Voxel block position corresponds to voxel block
  //! array(dev_voxel_block_array).
  My_Type::Vector3i *voxel_block_position;
  //! CUDA device pointer of voxel_block_position.
  My_Type::Vector3i *dev_voxel_block_position;

  //
  int min_depth, max_depth, *dev_min_depth, *dev_max_depth;
  //! Weight center of current depth frame's voxel block
  My_Type::Vector3f current_weight_center, *dev_current_weight_center;
  //! Weight center of Map's voxel block
  My_Type::Vector3f map_weight_center, *dev_map_weight_center;

#pragma region(Output, gengerated by raycast module)
  // --------------- For tracking ---------------
  //! Width/Height  of range_map.
  int raycast_range_map_width, raycast_range_map_height;
  //! Range map for raycast full resolution depth image.
  My_Type::Vector2f *raycast_range_map;
  My_Type::Vector2f *dev_raycast_range_map;

  //!
  int raycast_depth_width, raycast_depth_height;
  //! Raycast points
  My_Type::Vector3f *raycast_points;
  My_Type::Vector3f *dev_raycast_points;
  //! Raycast normal (caculate normal vector from SDF of voxel)
  My_Type::Vector3f *raycast_normal;
  My_Type::Vector3f *dev_raycast_normal;
  //! CUDA device pointer. Raycast points' fusiong weight.
  int *dev_raycast_weight;
  //! CUDA device pointer. Raycast points' plane label.
  int *dev_raycast_plane_label;

  // --------------- For view ---------------

  //! Width/Height of range_map.
  int scene_range_map_width, scene_range_map_height;
  //! Range map for raycast full resolution depth image.
  My_Type::Vector2f *scene_range_map;
  My_Type::Vector2f *dev_scene_range_map;

  //
  int scene_depth_width, scene_depth_height;
  //! Raycast scene points
  My_Type::Vector3f *scene_points;
  My_Type::Vector3f *dev_scene_points;
  //! CUDA device pointer. Raycast normal (caculate normal vector from SDF of
  //! voxel)
  My_Type::Vector3f *dev_scene_normals;
  //! CUDA device pointer. Raycast points' fusion weight.
  int *dev_scene_weight;
  //! CUDA device pointer. Raycast points' plane label.
  int *dev_scene_plane_label;
  ////! The color buffer array for (OpenGL) rendering scene_points.
  // My_Type::Vector4uc * scene_color;
  // My_Type::Vector4uc * dev_scene_color;
#pragma endregion

#pragma region(Debug)
  // for Debug
  char *allocate_flag;
  My_Type::Vector3i *ordered_position;
  int allocate_flag_counter;
#pragma endregion

  //! Initialize voxel map object.
  /*!
          \param	aligned_depth_size			Size of depth
     image for tracking

          \param	max_voxel_block_number

          \return	void
  */
  void init_Voxel_map(My_Type::Vector2i aligned_depth_size,
                      int max_voxel_block_number);

  //! Allocate voxel block by current frame points.
  /*!
          \param	dev_current_points	CUDA device pointer. Current
     depth frame points cloud.

          \param	camera_pose			The camera pose matrix in
     map coordinate.

          \param	submap_pose			The submap pose matrix
     in world coordinate.

          \return	int					Number of voxel
     block allocated.
  */
  int allocate_voxel_block(
      My_Type::Vector3f *dev_current_points, Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f submap_pose = Eigen::Matrix4f::Identity());

  //       Voxel
  //! Fusion SDF value to voxel with known camera pose and current points.
  /*!
          \param	dev_current_points	CUDA device pointer. Current
     depth frame points cloud.

          \param	dev_current_normal	CUDA device pointer. Current
     depth frame points cloud normal vectors.

          \param	camera_pose			The camera pose matrix in
     map coordinate.

          \return	void
  */
  void fusion_SDF_to_voxel(
      My_Type::Vector3f *dev_current_points,
      My_Type::Vector3f *dev_current_normal, Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f sumap_pose = Eigen::Matrix4f::Identity());
  //! Fusion plane label
  /*!
          \param	dev_current_points	CUDA device pointer. Current
     depth frame points cloud.

          \param	camera_pose			The camera pose matrix in
     map coordinate.

          \param	plane_img			CUDA device pointer.
     Current plane segmentation result.(A label mask image)

          \return	void
  */
  void fusion_plane_label_to_voxel(
      My_Type::Vector3f *dev_current_points, int *plane_img,
      Eigen::Matrix4f camera_pose,
      Eigen::Matrix4f sumap_pose = Eigen::Matrix4f::Identity());

  //! Merge fragment voxel map to this voxel map
  /*!
          \param	fragment_map			The fragment map object.

          \return	void
  */
  void merge_with_voxel_map(const Voxel_map &fragment_map,
                            int *dev_plane_global_index);

  //! Update from last voxel map (only update visible blocks)
  /*!

  */
  void update_from_last_voxel_map(My_pose &camera_pose,
                                  My_Type::Vector3f *dev_current_points,
                                  HashEntry *dev_entries,
                                  Voxel_f *dev_voxel_array);

  //! Raycast all information in voxel to several image matrices.
  /*!
          \param	camera_pose			The camera pose matrix in
     map coordinate.

          \param	mode				Raycast mode. See
     Raycast_flag.

          \return	void

          \note	Raycast is the inverse operation of projection. See details on
     wiki (https://en.wikipedia.org/wiki/Ray_casting).
  */
  void raycast_by_pose(Eigen::Matrix4f camera_pose, RaycastMode mode);

  //! Set map to empty statement.
  /*!
          \return	void
  */
  void clear_Voxel_map();

  //! Compress map
  void compress_voxel_map();

  //! Release voxel map
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

  //! Copy out data for debug or showing result.
  void extract_visible_list_to_CPU();

  //! Copy out data for debug or showing result.
  void copy_out_raycast_points();

  //! Reduce VoxelBlock's weight center
  void reduce_Voxel_map_weight_center();
};
