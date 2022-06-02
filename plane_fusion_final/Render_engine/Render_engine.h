/**
 *  @file Render_engine.h
 *  @brief Render engine, copy data needed from device to host.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include "Map_engine/voxel_definition.h"
#include "OurLib/Trajectory_node.h"
#include "Plane_detector/Plane_detector.h"
#include "Preprocess_engine/Hierarchy_image.h"
#include "SLAM_system/SLAM_system_settings.h"
#include "UI_engine/UI_parameters.h"
//#include "Map_engine/Plane_map.h"
//#include "Map_engine/Voxel_map.h"

/** @brief The MainViewportRenderMode enum. */
enum MainViewportRenderMode {
  /** Phong render */
  PHONG_RENDER = 0,
  /** SDF weight render */
  SDF_WEIGHT_RENDER = 1,
  /** Semantic plane render */
  SEMANTIC_PLANE_RENDER = 2,
  /** Number of enum */
  MVPRMODE_NUM = 3
};

/** @brief The NormalsSource enum. */
enum NormalsSource {
  /** Depth image normal */
  DEPTH_NORMAL = 0,
  /** Model normal */
  MODEL_NORMAL = 1,
};

/**
 * @brief The Render_engine class
 */
class Render_engine {
 public:
  /** @brief Depth image size. */
  My_Type::Vector2i depth_size;

  /** @brief Scene depth image size. */
  My_Type::Vector2i scene_depth_size;

  /** @brief Range map size. */
  My_Type::Vector2i range_map_size;

  /** @brief Ground truth trajectory. */
  Trajectory gound_truth_trajectory;
  /** @brief Estimated trajectory. */
  Trajectory estiamted_trajectory;

  /** @brief Current frame points. */
  My_Type::Vector3f *current_points;
  /** @brief Model points (generate by raycast_module). */
  My_Type::Vector3f *model_points;

  /** @brief Current hierarchy normals. */
  Hierarchy_image<My_Type::Line_segment> current_hierarchy_normal_to_draw;
  /** @brief Model hierarchy normals. */
  Hierarchy_image<My_Type::Line_segment> model_hierarchy_normal_to_draw;
  /** @brief Current hierarchy normals. */
  Hierarchy_image<My_Type::Line_segment> dev_current_hierarchy_normal_to_draw;
  /** @brief Model hierarchy normals. */
  Hierarchy_image<My_Type::Line_segment> dev_model_hierarchy_normal_to_draw;

  /** @brief Raycast range map (line segment). */
  My_Type::Vector2f *range_map;

  /** @brief Points for OpenGL rendering. */
  My_Type::Vector3f *scene_points;
  /** @brief Points for OpenGL rendering. */
  My_Type::Vector3f *dev_scene_points;

  /** @brief Normal vector of scene_points. */
  My_Type::Vector3f *scene_normals;
  /** @brief Normal vector of scene_points. */
  My_Type::Vector3f *dev_scene_normals;

  /** @brief ????. */
  My_Type::Vector3f *dev_scene_plane_label;

  /** @brief Weight of scene_points. */
  int *dev_scene_points_weight;

  /** @brief Color array of scene_points. */
  My_Type::Vector4uc *scene_points_color;
  /** @brief Color array of scene_points. */
  My_Type::Vector4uc *dev_scene_points_color;

  /** @brief Pseudo color array of plane labels. */
  My_Type::Vector4uc *pseudo_plane_color;
  /** @brief Pseudo color array of plane labels. */
  My_Type::Vector4uc *dev_pseudo_plane_color;

  /** @brief ???. */
  HashEntry *enties_buffer;

  /** @brief The number of voxel blocks. */
  int number_of_blocks = 0;
  /** @brief Voxel blocks' line. */
  My_Type::Vector3f *voxel_block_lines;

#if __unix__
#pragma region "Viewport2" {
#elif _WIN32
#pragma region(Viewport2)
#endif
  /** @brief millimeter. */
  int min_depth;
  /** @brief millimeter. */
  int *dev_min_depth;

  /** @brief millimeter. */
  int max_depth;
  /** @brief millimeter. */
  int *dev_max_depth;

  /** @brief Color buffer of viewport2. */
  My_Type::Vector4uc *viewport_2_color;
  /** @brief Color buffer of viewport2. */
  My_Type::Vector4uc *dev_viewport_2_color;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  /** @brief Default constructor. */
  Render_engine();
  /** @brief Default destructor. */
  ~Render_engine();

  /**
   * @brief Initialization
   * @param depth_size： Depth image size.
   * @param scene_depth_size Scene depth image size.
   */
  void init(My_Type::Vector2i depth_size, My_Type::Vector2i scene_depth_size);

  /**
   * @brief Scene viewport reshape function
   * @param scene_depth_size Scene depth image size.
   */
  void scene_viewport_reshape(My_Type::Vector2i scene_depth_size);

  /**
   * @brief Render scene points.
   * @param render_mode Render mode of scene points
   */
  void render_scene_points(MainViewportRenderMode render_mode);

  /**
   * @brief Render depth image with pseudo color.
   * @param dev_raw_aligned_points Raw points with alignement.
   */
  void pseudo_render_depth(My_Type::Vector3f *dev_raw_aligned_points);

  /**
   * @brief Render plane image with label.
   * @param dev_plane_labels
   */
  void pseudo_render_plane_labels(int *dev_plane_labels);

  /**
   * @brief Generate normal segment line for drawing.
   * @param dev_raw_aligned_points Raw points with alignment.
   * @param dev_normals Assigned normals.
   * @param normals_source From model or depth image.
   */
  void generate_normal_segment_line(My_Type::Vector3f *dev_raw_aligned_points,
                                    My_Type::Vector3f *dev_normals,
                                    NormalsSource normals_source);

  /**
   * @brief Generate voxel block line for drawing.
   * @param dev_entries HashEntry ??
   * @param number_of_entries
   */
  void generate_voxel_block_lines(
      HashEntry *dev_entries,
      int number_of_entries = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH));
};
