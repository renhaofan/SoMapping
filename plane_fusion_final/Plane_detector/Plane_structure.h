/**
 *  Copyright (C) All rights reserved.
 *  @file Plane_structure.h
 *  @brief Header file for plane map data structure.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include "OurLib/My_matrix.h"

/** @brief Min plane distance. */
#define MIN_PLANE_DISTANCE 0.02

/**
 * @brief Histogram step.
 * @note Try 0.02.
 */
#define HISTOGRAM_STEP 0.05

/**
 * @brief Histogram width.
 * @note 0.08 * 64 = 2.56 * 2 (-2.56 to +2.56)
 */
#define HISTOGRAM_WIDTH 128

/** @brief Maxmum of current planes. */
#define MAX_CURRENT_PLANES 64
/** @brief Maxmum of model planes. */
#define MAX_MODEL_PLANES 1024

/** @brief Unknown. */
#define MAX_HIST_NORMALS 1024

/** @brief Unknown. */
#define MAX_VALID_DEPTH_M 9.0

/** @brief Unknown. */
#define MIN_CELLS_OF_DIRECTION 55.0f
/** @brief Unknown. */
#define MIN_CELLS_OF_PLANE 50.0f

/** @brief Unknown. */
#define MIN_AREA_OF_DIRECTION 1.0f
/** @brief Unknown. */
#define MIN_AREA_OF_PLANE 0.5f

/** @brief Unknown. */
#define MIN_CELL_NORMAL_INNER_PRODUCT_VALUE 0.97

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

  /** @brief Unknown. */
  float weight;

  /** @brief The number of pixel the plane contained. */
  int pixel_number;

  /** @brief Plane ID. */
  int plane_index;

  /** @brief Unknown. */
  int global_index;

  /** @brief The number of cell the plane map contained. */
  int cell_num;

  /** @brief Area. */
  float area;

  /** @brief Unknown. Fragment Map Block. */
  int block_num;
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
        weight(_weight),
        plane_index(_plane_index),
        global_index(_global_index),
        cell_num(_cell_num),
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

/** @brief Size of super pixel CUDA block width. */
#define SUPER_PIXEL_BLOCK_WIDTH 16

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

/** @brief Plane Hash entries. */
#define ORDERED_PLANE_TABLE_LENGTH 0x40000
/** @brief Plane Hash entries. */
#define ORDERED_PLANE_TABLE_MASK 0x3FFFF

/** @brief Excess table length. */
#define EXCESS_PLANE_TABLE_LENGTH 0x10000

/** @brief Plane hash entry data structure. */
typedef struct PlaneHashEntry {
  /** @brief 2D position. */
  int position[2];
  // offset < 0 collision  offset >= 0  collision
  int offset;  //    int
  // ptr < 0   ptr >= 0
  int ptr;
} PlaneHashEntry;

/** @brief Plane pixel block width. */
#define PLANE_PIXEL_BLOCK_WIDTH 16
/** @brief Plane pixel block size. */
#define PLANE_PIXEL_BLOCK_SIZE \
  (PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_BLOCK_WIDTH)

/** @brief Track plane threshold. */
#define TRACK_PLANE_THRESHOLD 0.6
/** @brief Pixel block number */
#define PIXEL_BLOCK_NUM 0x10000

/**
 *  @brief Plane pixel size in meter.
 *  @note Try 0.005
 */
#define PLANE_PIXEL_SIZE 0.004
/** @brief Half of plane pixel size in meter. */
#define HALF_PLANE_PIXEL_SIZE PLANE_PIXEL_SIZE * 0.5
/** @brief Plane pixel block size in meter. */
#define PLANE_PIXEL_BLOCK_WIDTH_M (PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_SIZE)

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
