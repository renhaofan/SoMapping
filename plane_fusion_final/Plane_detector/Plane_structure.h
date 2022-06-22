/**
 *  @file Plane_structure.h
 *  @brief Header file for plane map data structure.
 *  @details Parameterize plane equaction \f$n_{x}x + n_{y}y + n_{z}z + d = 0\f$
 * by a <b>plane vector</b> \f$ \pi = (n_{x}, n_{y}, n_{z}, d) \f$,
 *
 *  where \f$ (n_{x}, n_{y}, n_{z}) \f$ is the <b>unit normal vector</b> of the
 * plane, so \f$ d \f$ is the distance from the plane to the origin.
 *
 *  Besides, take the closet point from the world origin to the plane as the
 * <b>plane coordinate origin</b>.
 *
 * Last but not least, When plane map was created, establish the plane vector.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include "OurLib/My_matrix.h"
#include "Config.h"

/** @brief Cell info data structure. */
typedef struct struct_Cell_info {
  /** @brief x component of cell points normal vector. */
  float nx;
  /** @brief y component of cell points normal vector. */
  float ny;
  /** @brief z component of cell points normal vector. */
  float nz;
  /** @brief x component of cell points position. */
  float x;
  /** @brief y component of cell points position. */
  float y;
  /** @brief z component of cell points position. */
  float z;
  /** @brief Unknown. */
  int plane_index;
  /** @brief Unknown. */
  int counter;
  /** @brief Area of this super pixel (m^2). */
  float area;
  /** @brief Unknown. */
  bool is_valid_cell;
} Cell_info;

/**
 *  @brief Plane info data structure.
 *  @details Parameterize plane equaction \f$n_{x}x + n_{y}y + n_{z}z + d = 0\f$
 * by a <b>plane vector</b> \f$ \pi = (n_{x}, n_{y}, n_{z}, d) \f$,
 *
 *  where \f$ (n_{x}, n_{y}, n_{z}) \f$ is the <b>unit normal vector</b> of the
 * plane, so \f$ d \f$ is the distance from the plane to the origin.
 *
 *  Besides, take the closet point from the world origin to the plane as the
 * <b>plane coordinate origin</b>.
 *
 * Last but not least, When plane map was created, establish the plane vector.
 */
typedef struct struct_Plane_info {
  /** @brief x component of plane coordinate orgin. */
  float x;
  /** @brief y component of plane coordinate orgin. */
  float y;
  /** @brief z component of plane coordinate orgin. */
  float z;

  /** @brief x component of plane vector. */
  float nx;
  /** @brief y component of plane vector. */
  float ny;
  /** @brief z component of plane vector. */
  float nz;
  /** @brief The distance from the plane to the origin. */
  float d;

  /** @brief The number of pixel the plane contained. */
  int pixel_num;
  /** @brief The number of cell the plane map contained. */
  int cell_num;
  /** @brief Unknown. Fragment Map Block. */
  int block_num;

  /** @brief Plane ID. */
  int plane_index;
  /** @brief Unknown. */
  int global_index;

  /** @brief Unknown. */
  float weight;
  /** @brief Area. */
  float area;

  /** @brief Unknown. */
  bool is_valid;

  /** @brief Default constructor. */
  struct_Plane_info(){};
  /** @brief Init constructor. */
  struct_Plane_info(float _nx, float _ny, float _nz, float _weight,
                    int _plane_index, int _cell_num, bool _is_valid,
                    int _global_index)
      : nx(_nx),
        ny(_ny),
        nz(_nz),
        cell_num(_cell_num),
        plane_index(_plane_index),
        global_index(_global_index),
        weight(_weight),
        is_valid(_is_valid) {}

} Plane_info;

/** @brief Hist normal data structure. */
typedef struct struct_Hist_normal {
  float nx, ny, nz;
  // Patch
  int counter;
  float weight;
} Hist_normal;

/** @brief Plane match data structure. */
typedef struct struct_Plane_match_info {
  int current_plane_id;
  int model_plane_id;
  // Patch
  float match_weight;
} Plane_match_info;

/** @brief Superpixel data structure. */
typedef struct struct_Super_pixel {
  /** @brief 2D position. */
  union {
    float center_data[2];
    struct {
      float cx, cy;
    };
  };

  /** @brief Normal vector. */
  union {
    float normal_data[3];
    struct {
      float nx, ny, nz;
    };
  };

  /** @brief 3D position */
  union {
    float position_data[3];
    struct {
      float px, py, pz;
    };
  };

  /** @brief The number of valid pixels. */
  int valid_pixel_number;
  /** @brief Flag whether this super pixel is plannar cell. */
  bool is_planar_cell;

} Super_pixel;


/** @brief Plane hash entry data structure. */
typedef struct PlaneHashEntry {
  /** @brief 2D position. */
  int position[2];
  // offset < 0 collision  offset >= 0  collision
  int offset;  //    int
  // ptr < 0   ptr >= 0
  int ptr;
} PlaneHashEntry;


/** @brief Plane pixel data structure. */
typedef struct struct_Plane_PIXEL {
  /**
   * @brief diff deviation distance from the real surface to plane.
   * @details Positive: outside of the surface.
   * @details Negative: inside of the surface.
   */
  float diff;
  /**
   * @brief Label of plane (id in fact).
   * @details 0 meas not a plane.
   */
  int plane_label;
} Plane_pixel;

/** @brief Plane coordinate data structure. */
typedef struct struct_Plane_coordinate {
  /** @brief Unknown. */
  My_Type::Vector3f x_vec;
  /** @brief Unknown. */
  My_Type::Vector3f y_vec;
  /** @brief Unknown. */
  My_Type::Vector3f z_vec;
} Plane_coordinate;
