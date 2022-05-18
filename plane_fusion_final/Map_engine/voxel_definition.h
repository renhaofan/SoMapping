

#pragma once

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
#define SUBMAP_VOXEL_BLOCK_NUM 0x008000
//#define SUBMAP_VOXEL_BLOCK_NUM	0x00A000
//#define SUBMAP_VOXEL_BLOCK_NUM	0x00F000
//#define SUBMAP_VOXEL_BLOCK_NUM		0x008000
//#define SUBMAP_VOXEL_BLOCK_NUM		0x017000

//
#define MAX_SDF_WEIGHT_MASK 0x7F
//   Block  Voxel
#define VOXEL_BLOCK_WDITH 8
#define VOXEL_BLOCK_SIZE                                                       \
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

//   Voxel
// float  voxel
typedef struct _Voxel_f {
  float sdf;
  unsigned char color[3];
  unsigned char weight;
  unsigned short plane_index;
} Voxel_f;

// Hash
typedef struct _HashEntry {
  //   Block     (（  ：   ）
  int position[3];
  //   hash collision ，   excess entries  HashEntry
  // offset < 0  collision； offset >= 0  collision
  int offset; //    int
  //   Block
  // ptr < 0    ； ptr >= 0
  int ptr;

} HashEntry;
