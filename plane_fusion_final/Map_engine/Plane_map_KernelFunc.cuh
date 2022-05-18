/*!

        \file		Voxel_map_KernelFunc.cuh

        \brief		Voxel kernel functions

        \details

        \author		GongBingjian

        \date		2018-06-01

        \version	V2.0

        \par	Copyright (c):
        2018-2019 GongBingjian All rights reserved.

        \par	history
        2019-03-09 16:26:40	Doc by Doxygen

*/

// voxel
#include "SLAM_system/SLAM_system_settings.h"
#include "voxel_definition.h"

// My type
#include "OurLib/My_matrix.h"
#include "Plane_detector/Plane_structure.h"

//!
/*!


*/
void build_allocate_flag_CUDA(dim3 block_rect, dim3 thread_rect,
                              const HashEntry *allocated_entries,
                              const Voxel_f *voxel_block_array,
                              const PlaneHashEntry *plane_entries,
                              Plane_coordinate plane_coordinate,
                              int model_plane_id, bool *need_allocate,
                              My_Type::Vector2i *allocate_pixel_block_pos);

//!
/*!

*/
void allocate_plane_blocks_CUDA(
    dim3 block_rect, dim3 thread_rect, bool *need_allocate,
    const My_Type::Vector2i *allocate_pixel_block_pos,
    PlaneHashEntry *plane_entries, int *excess_counter, int *number_of_blocks);

//!
/*!

*/
void find_allocated_planar_entries_CUDA(dim3 block_rect, dim3 thread_rect,
                                        const PlaneHashEntry *entries,
                                        int *index_array,
                                        int *allocated_counter);

//!
/*!

*/
void fusion_sdf_to_plane_CUDA(
    dim3 block_rect, dim3 thread_rect, const HashEntry *allocated_entries,
    const Voxel_f *voxel_block_array, const int *entries_index_list,
    Plane_info model_plane, Plane_coordinate plane_coordinate,
    PlaneHashEntry *plane_entries, Plane_pixel *plane_pixel_array);

//!
/*!

*/
void generate_block_vertex_CUDA(dim3 block_rect, dim3 thread_rect,
                                const PlaneHashEntry *entries,
                                Plane_coordinate plane_coordinate,
                                Plane_info model_plane,
                                My_Type::Vector3f *block_vertexs,
                                int *number_of_blocks);
