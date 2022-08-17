#pragma once
#include "Config.h"

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
  int offset;  //    int
  //   Block
  // ptr < 0    ； ptr >= 0
  int ptr;

} HashEntry;
