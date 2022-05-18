/*!
        \file		Track_KernelFunc.cuh

        \brief		Defination of Track_Engine CUDA kernel functions.

        \details

        \author		GongBingjian

        \date		2018-04-10

        \version	V2.0

        \par	Copyright (c):
        2018-2019 GongBingjian All rights reserved.

        \par	history
        2019-03-11 13:25:07		Doc by Doxygen

*/

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <math.h>

// arg
#include "OurLib/My_matrix.h"
#include "SLAM_system/SLAM_system_settings.h"
#include "Track_structure.h"

// Point-to-Plane ICP residual
void compute_points_residual_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, const float *points_weight,
    Sensor_params sensor_param, My_Type::Matrix44f incremental_pose,
    int layer_id, Accumulate_result *accumulate_result);
//
void generate_correspondence_lines_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, Sensor_params sensor_param,
    My_Type::Matrix44f incremental_pose,
    My_Type::Vector3f *correspondence_lines);

// Photometric residual
void compute_photometric_residual_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const float *current_intensity, const float *model_intensity,
    const My_Type::Vector2f *model_gradient, const float *points_weight,
    Sensor_params sensor_param, My_Type::Matrix44f incremental_pose,
    int layer_id, Accumulate_result *accumulate_result);
