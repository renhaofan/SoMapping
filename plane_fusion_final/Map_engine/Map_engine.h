#pragma once

//
#include "OurLib/My_pose.h"
#include "OurLib/My_vector.h"

//
#include "Map_engine/Feature_map.h"
#include "Map_engine/Plane_map.h"
#include "Map_engine/Voxel_map.h"

//! Map engine interface
/*!

*/
class Map_engine {
public:
  //! Model points position for tracking
  My_Type::Vector3f *dev_model_points;
  //! Model points position for tracking
  My_Type::Vector3f *dev_model_normals;
  //! Model points weight for tracking
  int *dev_model_weight;
  //! Model plane label for tracking
  int *dev_model_plane_labels;

  //! Model points position for view
  My_Type::Vector3f *scene_points;
  //! Model points position for view
  My_Type::Vector3f *scene_normals;
  //! Model points weight for view
  int *scene_weight;
  //! Model plane label for view
  int *scene_plane_labels;

  //! Constructor/Destructor
  virtual ~Map_engine();

  //!
  virtual void init_map() = 0;

  //! Initialization
  void init_base();

  //! Change buffer size when render viewport reshaped.
  void reshape_render_viewport(My_Type::Vector2i viewport_size);

  //! Update map after tracking done
  virtual void update_map_after_tracking(My_pose &camera_pose,
                                         My_Type::Vector3f *dev_current_points,
                                         My_Type::Vector3f *dev_current_normals,
                                         int *dev_plane_labels = nullptr) = 0;

  // Update plane map
  virtual void update_plane_map(const Plane_info *current_planes,
                                std::vector<My_Type::Vector2i> &matches) = 0;

  //! Update map after optimi done
  virtual void update_map_after_optimization() = 0;

  //! Generate points cloud of single view from map
  virtual void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                                       RaycastMode raycast_mode) = 0;
};

//!
/*!

*/
class Blank_map : public Map_engine {
public:
  Blank_map(){};
  ~Blank_map(){};

private:
  //
  void init_map() override { init_base(); }

  //! Update map after tracking done
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override{};

  //! Update map after optimi done
  void update_map_after_optimization() override{};

  //
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override{};

  //! Generate points cloud of single view from map
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override{};
};

//! Voxel based map
/*!

*/
class Basic_Voxel_map : public Map_engine {
public:
  //!
  bool voxel_map_fusion_initialize_done = false;
  //! Basic voxel map
  Voxel_map *voxel_map_ptr;
  //! Plane map
  Plane_map *plane_map_ptr;

  //! Constructor/Destructor
  Basic_Voxel_map();
  ~Basic_Voxel_map();

  //!
  void init_map() override;

  //! Update map after tracking done
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override;

  //!
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override;

  //!
  void generate_plane_map();

  //! Generate points cloud of single view from map
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override;

  //!
  void update_map_after_optimization() override{};
};

//! Submap voxel map
/*!

*/
class Submap_Voxel_map : public Map_engine {
public:
  //! Voxel map
  Voxel_map *voxel_map_ptr;
  //! Plane map
  Plane_map *plane_map_ptr;

  //! Flag for constant map
  bool map_completed = false;
  //!
  int init_number_of_blocks;
  //
  bool first_frame_to_this_submap = true;
  //
  int frame_counter = 0;

  //! Constructor/Destructor
  Submap_Voxel_map();
  ~Submap_Voxel_map();

  //!
  void init_map() override;

  //! Update from last map
  void update_map_form_last_map(My_pose &camera_pose,
                                My_Type::Vector3f *dev_current_points,
                                HashEntry *dev_entries,
                                Voxel_f *dev_voxel_array);

  //! Update map after tracking done
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override;

  //
  //!
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override;

  //!
  void generate_plane_map();

  //! Generate points cloud of single view from map
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override;

  //!
  void update_map_after_optimization() override{};

  //!
  bool consider_to_create_new_submap();

  //!
  void compress_voxel_map();

  //!
  void release_voxel_map();
};
