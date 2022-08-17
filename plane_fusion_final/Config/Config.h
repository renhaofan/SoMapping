/**
 *  Copyright (C) All rights reserved.
 *  @file Config.h
 *  @brief Define some parameters and macros.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @todo void init();
 *  @todo void check();  check confilct and valid values, include only one slam
 * system set up true.
 *  @warning Current ctor, load json file once. need to finish function init().
 */

#pragma once

#include "json.hpp"
using json = nlohmann::json;

#include "Log.h"

// file operation
#include <dirent.h>
#ifdef _WIN32
#include <io.h>
#elif __linux__
#include <inttypes.h>
#include <unistd.h>
#define __int64 int64_t
#define _close close
#define _read read
#define _lseek64 lseek64
#define _O_RDONLY O_RDONLY
#define _open open
#define _lseeki64 lseek64
#define _lseek lseek
#define stricmp strcasecmp
#endif
#include <fstream>
#include <string>
#include <vector>

/**
 *  @brief Maximum number of hierarchy layers.
 *  @details Used directly by Hierarchy_image.h, SLAM_system_settings.h
 */
#define MAX_LAYER_NUMBER 8

/**
 *  @brief Flag whether enable SLAM debug code for
 *  @details Used directly by SLAM_system_settings.h
 */
#define COMPILE_DEBUG_CODE


#if __unix__
#pragma region "Plane structure.h used" {
#elif _WIN32
#pragma region(Plane structure)
#endif

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

/** @brief Plane Hash entries. */
#define ORDERED_PLANE_TABLE_LENGTH 0x40000
/** @brief Plane Hash entries. */
#define ORDERED_PLANE_TABLE_MASK 0x3FFFF

/** @brief Excess table length. */
#define EXCESS_PLANE_TABLE_LENGTH 0x10000

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

/** @brief Size of super pixel CUDA block width. */
#define SUPER_PIXEL_BLOCK_WIDTH 16

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "voxel_definition.h used" {
#elif _WIN32
#pragma region(voxel_definition.h)
#endif
// Voxel Space
//
//
#define ORDERED_TABLE_LENGTH 0x100000
#define ORDERED_TABLE_MASK 0x0FFFFF
//      （  Collosion）
#define EXCESS_TABLE_LENGTH 0x010000
//      Block
#define VISIBLE_LIST_LENGTH 0x040000

// VoxelBlock
//        VoxelBlock
#define VOXEL_BLOCK_NUM 0x017000
//        VoxelBlock
#define SUBMAP_VOXEL_BLOCK_NUM 0x09000
//#define SUBMAP_VOXEL_BLOCK_NUM	0x00A000
//#define SUBMAP_VOXEL_BLOCK_NUM	0x00F000
//#define SUBMAP_VOXEL_BLOCK_NUM		0x008000
//#define SUBMAP_VOXEL_BLOCK_NUM		0x017000

//
#define MAX_SDF_WEIGHT_MASK 0x7F
//   Block  Voxel
#define VOXEL_BLOCK_WDITH 8
#define VOXEL_BLOCK_SIZE \
(VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH)
//   Voxel  				5mm
//#define VOXEL_SIZE				0.008f
//#define VOXEL_SIZE				0.006f
#define VOXEL_SIZE 0.01f
//#define VOXEL_SIZE				0.005f
//#define VOXEL_SIZE				0.0075f
#define HALF_VOXEL_SIZE VOXEL_SIZE * 0.5f
#define VOXEL_SIZE_MM (VOXEL_SIZE * 1000.0f)

//     （5  ）
//#define TRUNCATED_BAND		0.08f
#define TRUNCATED_BAND 0.04f
//#define TRUNCATED_BAND			0.02f
#define TRUNCATED_BAND_MM (TRUNCATED_BAND * 1000.0f)
//     （128）
#define MAX_SDF_WEIGHT 100

// DDA
//   DDA    （  ： ）
#define DDA_STEP_SIZE (VOXEL_SIZE * VOXEL_BLOCK_WDITH)
#define DDA_STEP_SIZE_MM (VOXEL_SIZE_MM * VOXEL_BLOCK_WDITH)

//
#define PRIME_X 73856093u
#define PRIME_Y 19349669u
#define PRIME_Z 83492791u

#define INVISIBLE_BLOCK 0
#define VISIBLE_BLOCK 1
#define NOT_NEED_ALLOCATE 0
#define NEED_ALLOCATE 1

#define MIN_RAYCAST_WEIGHT 2

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif


class JSON_CONFIG {
 public:
  static JSON_CONFIG* instance_ptr;
  static JSON_CONFIG* instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new JSON_CONFIG();
    // check whether allocte memory succesfully.
    if (instance_ptr == nullptr) {
#ifdef LOGGING
      LOG_FATAL("Failed to allocate JSON CONFIG memory!");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(): "
              "Failed to allocate JSON CONFIG memory.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Failed to allocate JSON CONFIG memory!";
    }
    return instance_ptr;
  }
  json j;

  void init();
  void check();

  JSON_CONFIG();
  ~JSON_CONFIG();
};
