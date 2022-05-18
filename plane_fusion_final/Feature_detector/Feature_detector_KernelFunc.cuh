

// My type
#include "OurLib/My_matrix.h"
//
#include "SLAM_system/SLAM_system_settings.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//
void convert_intensity_to_cvmat_KernelFunc(dim3 block_rect, dim3 thread_rect,
                                           const float *src_intensity,
                                           My_Type::Vector2i image_size,
                                           My_Type::Vector2f intensity_range,
                                           unsigned char *dst_intensity);
