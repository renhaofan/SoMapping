/*!

        \file		Voxel_KernelFunc.cuh

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
//#include "voxel_defination.h"
#include "voxel_defination.h"

// My type.  Include in SLAM_system_settings.h
//#include "OurLib/My_matrix.h"

//        block   entry flags
void build_entry_flag_CUDA(dim3 block_rect, dim3 thread_rect,
                           const My_Type::Vector3f *dev_current_points,
                           const float *map_pose, const HashEntry *entry,
                           char *allocate_flag, My_Type::Vector3i *order_pos);
//--  From submap
void build_entry_flag_CUDA(dim3 block_rect, dim3 thread_rect,
                           My_Type::Vector3i *block_position, int block_num,
                           float *pose, HashEntry *entry, char *allocate_flag,
                           My_Type::Vector3i *order_pos);

//   entry flags     Block    Entry
void allocate_by_flag_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry *entry,
                           const char *allocate_flag,
                           My_Type::Vector3i *pos_buffer, int *excess_counter,
                           int *number_of_blocks,
                           My_Type::Vector3i *voxel_block_position);

//        block   entry flags
void build_visible_flag_CUDA(dim3 block_rect, dim3 thread_rect,
                             const My_Type::Vector3f *current_points,
                             const float *camera_pose, HashEntry *entry,
                             char *visible_flag);
//--  From submap
void build_relative_flag_CUDA(dim3 block_rect, dim3 thread_rect,
                              My_Type::Vector3i *block_position, int block_num,
                              float *pose, HashEntry *entry,
                              char *relative_flag);

//   entry flags       VisbleList(entry)
void build_visible_list_CUDA(dim3 block_rect, dim3 thread_rect,
                             const HashEntry *entry, HashEntry *visible_list,
                             const char *visible_flag, int *visible_counter,
                             const float *pose_inv, int *min_depth,
                             int *max_depth);
//--  From submap
void build_relative_list_CUDA(dim3 block_rect, dim3 thread_rect,
                              HashEntry *entry, HashEntry *relative_list,
                              char *relative_flag, int *relative_counter,
                              float *pose_inv);

//   Raycast Range
void raycast_get_range_CUDA(dim3 block_rect, dim3 thread_rect,
                            const float *view_pose, const HashEntry *entry,
                            Sensor_params sensor_params,
                            int raycast_patch_width, const int *min_distance,
                            const int *max_distance,
                            My_Type::Vector2f *range_map);
void raycast_get_range_4corner_CUDA(
    dim3 block_rect, dim3 thread_rect, const float *view_pose,
    Sensor_params sensor_params, int raycast_patch_width,
    const HashEntry *entry, const int *min_distance, const int *max_distance,
    My_Type::Vector2f *range_map);

// Raycast     （  Range   Step    ）
//
void raycast_byStep_CUDA(dim3 block_rect, dim3 thread_rect,
                         const float *__restrict__ camera_pose,
                         Sensor_params sensor_params, int raycast_patch_width,
                         const HashEntry *__restrict__ entry,
                         const Voxel_f *__restrict__ voxel_block_array,
                         const My_Type::Vector2f *__restrict__ range_map,
                         My_Type::Vector3f *raycast_points,
                         My_Type::Vector3f *raycast_normal,
                         int *raycast_weight);
//       （  ）
void raycast_byStep_CUDA(dim3 block_rect, dim3 thread_rect,
                         const float *camera_pose, Sensor_params sensor_params,
                         int raycast_patch_width, const HashEntry *entry,
                         const Voxel_f *voxel_block_array,
                         const My_Type::Vector2f *range_map,
                         My_Type::Vector3f *raycast_points,
                         My_Type::Vector3f *raycast_normal,
                         int *raycast_plane_label, int *raycast_weight);

//           voxel
//
void prj_fusion_sdf_CUDA(dim3 block_rect, dim3 thread_rect,
                         const My_Type::Vector3f *current_points,
                         const float *camera_pose_inv,
                         Sensor_params sensor_params, int depth_width,
                         int depth_height, const HashEntry *visible_list,
                         Voxel_f *voxel_block_array);
// with normal
void prj_normal_fusion_sdf_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *current_points,
                                const My_Type::Vector3f *current_normal,
                                const float *camera_pose_inv,
                                Sensor_params sensor_params, int depth_width,
                                int depth_height, const HashEntry *visible_list,
                                Voxel_f *voxel_block_array);
//   ：
void prj_fusion_plane_label_CUDA(dim3 block_rect, dim3 thread_rect,
                                 const My_Type::Vector3f *current_points,
                                 const float *camera_pose_inv,
                                 Sensor_params sensor_params, int depth_width,
                                 int depth_height,
                                 const HashEntry *visible_list,
                                 Voxel_f *voxel_block_array, int *plane_img);

// Fragment Map to Global Map merge
void fusion_map_sdf_CUDA(dim3 block_rect, dim3 thread_rect,
                         HashEntry *relative_list, int *relative_counter,
                         Voxel_f *this_Voxel_map, float *pose_inv,
                         HashEntry *fragment_entry, Voxel_f *fragment_Voxel_map,
                         int *plane_global_index);

//   Map        、  Block
void reduce_range_CUDA(dim3 block_rect, dim3 thread_rect,
                       const My_Type::Vector3i *__restrict__ block_position,
                       int block_num, int *min_depth, int *max_depth,
                       const float *pose_inv);

//   GlobalMap RaycastPoints      FragmentMap RaycastPoints
void merge_frame_with_occlusion_CUDA(
    dim3 block_rect, dim3 thread_rect, My_Type::Vector3f *global_points,
    My_Type::Vector3f *fragment_points, My_Type::Vector3f *merge_points,
    My_Type::Vector3f *global_normal, My_Type::Vector3f *fragment_normal,
    My_Type::Vector3f *merge_normal, int *global_weight, int *fragment_weight,
    int *merge_weight, int *global_plane_img, int *fragment_plane_img,
    int *merge_plane_img);

//     VoxelMap
void copy_voexl_map_CUDA(dim3 block_rect, dim3 thread_rect,
                         HashEntry *relative_list, int *relative_counter,
                         Voxel_f *dst_Voxel_map, HashEntry *fragment_entry,
                         Voxel_f *fragment_Voxel_map);

// Reduce the weight center of current voxel block
void reduce_current_voxel_block_position_CUDA(
    dim3 block_rect, dim3 thread_rect, HashEntry *visible_list,
    My_Type::Vector3f *current_weight_center, int visible_counter);

// Reduce the (weight center/min point/max point) of Map's voxel block
void reduce_map_voxel_block_position_CUDA(dim3 block_rect, dim3 thread_rect,
                                          HashEntry *entry,
                                          My_Type::Vector3f *map_weight_center);

// Reduce the bunding box (int offset) of Map's Voxel block
void reduce_map_bounding_box_CUDA(dim3 block_rect, dim3 thread_rect,
                                  HashEntry *entry,
                                  My_Type::Vector3i *map_min_offset,
                                  My_Type::Vector3i *map_max_offset);
