

#include "Plane_detector/Plane_structure.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//
#ifndef PRIME_X
#define PRIME_X 73856093u
#endif
#ifndef PRIME_Y
#define PRIME_Y 19349669u
#endif

//   hash
__device__ inline int plane_hash_func(int &x, int &y) {
  return (((((unsigned int)x) * PRIME_X) ^ (((unsigned int)y) * PRIME_Y)) &
          (unsigned int)ORDERED_PLANE_TABLE_MASK);
}

//
__device__ inline int get_pixel_index(int u_offset, int v_offset,
                                      const PlaneHashEntry *plane_entries) {
  //
  bool find_pixel = false;
  //
  int block_x = floorf((float)u_offset / (float)PLANE_PIXEL_BLOCK_WIDTH);
  int block_y = floorf((float)v_offset / (float)PLANE_PIXEL_BLOCK_WIDTH);
  //
  int entry_index = plane_hash_func(block_x, block_y);
  PlaneHashEntry temp_entry;
  do {
    temp_entry = plane_entries[entry_index];

    if (temp_entry.position[0] == block_x &&
        temp_entry.position[1] == block_y) {
      find_pixel = true;
    }

    entry_index = temp_entry.offset;
  } while (!find_pixel && entry_index > 0);

  //
  if (find_pixel) {
    //
    int offset_in_block_x = u_offset - block_x * PLANE_PIXEL_BLOCK_WIDTH;
    int offset_in_block_y = v_offset - block_y * PLANE_PIXEL_BLOCK_WIDTH;

    return (temp_entry.ptr + offset_in_block_x +
            offset_in_block_y * PLANE_PIXEL_BLOCK_WIDTH);
  } else {
    return -1;
  }
}
