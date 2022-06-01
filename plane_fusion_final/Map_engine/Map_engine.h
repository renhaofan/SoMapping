/**
 *  Copyright (C) All rights reserved.
 *  @file Map_engine.h
 *  @brief Map engine,
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include "Map_engine/Feature_map.h"
#include "Map_engine/Plane_map.h"
#include "Map_engine/Voxel_map.h"
#include "OurLib/My_pose.h"
#include "OurLib/My_vector.h"

/**
 * @brief Map base class.
 */
class Map_engine {
 public:
  /** @brief Model points position for tracking. */
  My_Type::Vector3f *dev_model_points;
  /** @brief Model points' normal vector for tracking. */
  My_Type::Vector3f *dev_model_normals;

  /** @brief Model points weight for tracking. */
  int *dev_model_weight;
  /** @brief Model plane label for tracking. */
  int *dev_model_plane_labels;

  /** @brief Model points position for view. */
  My_Type::Vector3f *scene_points;
  /** @brief Model points position for view. */
  My_Type::Vector3f *scene_normals;
  /** @brief Model points weight for view. */
  int *scene_weight;
  /** @brief Model plane label for view. */
  int *scene_plane_labels;

  /** @brief Default destructor. Release CPU & GPU memory. */
  virtual ~Map_engine();

  /** @brief Init plane map. */
  virtual void init_map() = 0;

  /** @brief Compute and allocate CPU & GPU memory for member variables. */
  void init_base();

  /** @brief Change buffer size when render viewport reshaped. */
  void reshape_render_viewport(My_Type::Vector2i viewport_size);

  /** @brief Update map after tracking done. */
  virtual void update_map_after_tracking(My_pose &camera_pose,
                                         My_Type::Vector3f *dev_current_points,
                                         My_Type::Vector3f *dev_current_normals,
                                         int *dev_plane_labels = nullptr) = 0;

  /** @brief Update plane map. */
  virtual void update_plane_map(const Plane_info *current_planes,
                                std::vector<My_Type::Vector2i> &matches) = 0;

  /** @brief Update map after optimi done. */
  virtual void update_map_after_optimization() = 0;

  /** @brief Generate points cloud of single view from map. */
  virtual void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                                       RaycastMode raycast_mode) = 0;
};

/**
 * @brief The Blank_map class
 */
class Blank_map : public Map_engine {
 public:
  /** @brief Default constructor. */
  Blank_map(){};
  /** @brief Default destructor. */
  ~Blank_map(){};

 private:
  /** @brief Initizalize map CPU & GPU data handler. */
  void init_map() override { Map_engine::init_base(); }

  /** @brief Update map after tracking done. */
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override{};

  /** @brief Update map after optimi done. */
  void update_map_after_optimization() override{};

  /** @brief Update plane map. */
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override{};

  /** @brief Generate points cloud of single view from map. */
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override{};
};

/**
 * @brief Voxel based map
 */
class Basic_Voxel_map : public Map_engine {
 public:
  /** @brief Flag whether voxel map fusion ready. */
  bool voxel_map_fusion_initialize_done = false;
  /** @brief Basic voxel map. */
  Voxel_map *voxel_map_ptr;
  /** @brief Plane map. */
  Plane_map *plane_map_ptr;

  /** @brief Default constructor, instantiate voxel map and plane map. */
  Basic_Voxel_map();
  /** @brief Default destructor. */
  ~Basic_Voxel_map();

  /** @brief Initialize voxel map and plane map. */
  void init_map() override;

  /**
   * @brief Update voxel map after tracking done.
   * @param camera_pose
   * @param dev_current_points
   * @param dev_current_normals
   * @param dev_plane_labels
   */
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override;

  /**
   * @brief update_plane_map
   * @param current_planes
   * @param matches
   */
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override;

  /**
   * @brief generate_plane_map
   */
  void generate_plane_map();

  /**
   * @brief Generate points cloud of single view from map.
   * @param camera_pose
   * @param raycast_mode
   */
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override;

  /**
   * @brief update_map_after_optimization
   */
  void update_map_after_optimization() override{};
};

/**
 * @brief Submap voxel map
 */
class Submap_Voxel_map : public Map_engine {
 public:
  /** @brief Voxel map. */
  Voxel_map *voxel_map_ptr;
  /** @brief Plane map. */
  Plane_map *plane_map_ptr;

  /** @brief Flag for constant map. */
  bool map_completed = false;

  /** @brief Unknown. */
  int init_number_of_blocks;

  /** @brief Unknown. */
  bool first_frame_to_this_submap = true;

  /** @brief Unknown. */
  int frame_counter = 0;

  /** @brief Default constructor. */
  Submap_Voxel_map();
  /** @brief Default destructor. */
  ~Submap_Voxel_map();

  /**
   * @brief init_map
   */
  void init_map() override;

  /**
   * @brief Update from last map.
   * @param camera_pose
   * @param dev_current_points
   * @param dev_entries
   * @param dev_voxel_array
   */
  void update_map_form_last_map(My_pose &camera_pose,
                                My_Type::Vector3f *dev_current_points,
                                HashEntry *dev_entries,
                                Voxel_f *dev_voxel_array);

  /**
   * @brief Update map after tracking done.
   * @param camera_pose
   * @param dev_current_points
   * @param dev_current_normals
   * @param dev_plane_labels
   */
  void update_map_after_tracking(My_pose &camera_pose,
                                 My_Type::Vector3f *dev_current_points,
                                 My_Type::Vector3f *dev_current_normals,
                                 int *dev_plane_labels = nullptr) override;

  /**
   * @brief update_plane_map
   * @param current_planes
   * @param matches
   */
  void update_plane_map(const Plane_info *current_planes,
                        std::vector<My_Type::Vector2i> &matches) override;

  /** @brief Generate plane points. */
  void generate_plane_map();

  /** @brief Generate points cloud of single view from map. */
  void raycast_points_from_map(Eigen::Matrix4f &camera_pose,
                               RaycastMode raycast_mode) override;

  /** @brief .... */
  void update_map_after_optimization() override{};

  /** @brief .... */
  bool consider_to_create_new_submap();

  /** @brief .... */
  void compress_voxel_map();

  /** @brief .... */
  void release_voxel_map();
};
