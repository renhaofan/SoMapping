
// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>


//
#include "Voxel_KernelFunc.cuh"

#include <float.h>


// 
__device__ inline float floor_by_stride(float & value, float & step);
// 
__device__ inline float round_by_stride(float & value, float & step);
// 
__device__ inline float ceil_by_stride(float & value, float & step);
// 
__device__ inline int hash_func(int & x, int & y, int & z);


// 
__device__ inline bool check_block_allocated(float & Vx, float & Vy, float & Vz, const HashEntry * entry);
//
__device__ inline bool check_block_allocated(int & block_pos_x, int & block_pos_y, int & block_pos_z, const HashEntry * entry);
// 
__device__ inline int find_voxel_index_round(float & Vx, float & Vy, float & Vz, const HashEntry * entry);
__device__ inline int get_voxel_index_neighbor(float & Vx, float & Vy, float & Vz, const HashEntry * entry);
// SDF 
__device__ inline bool get_sdf_interpolated(float & Vx, float & Vy, float & Vz, 
											const HashEntry * entry, const Voxel_f * voxel_block_array, 
											float & sdf_interpolated);
// Compute normal vector from 8 neighbor voxel
__device__ inline bool interpolate_normal_by_sdf(float & Vx, float & Vy, float & Vz,
												 const HashEntry * entry, const Voxel_f * voxel_block_array,
												 My_Type::Vector3f & normal_vector);


// Block of 256 threads Reduce
template<typename T>
inline __device__ void block_256_reduce(volatile T * cache_T, int tid);


// Warp Reduce minimum
template<typename T>
inline __device__ void warp_reduce_min(volatile T * cache_T, int tid);
// Block of 256 threads Reduce minimum
template<typename T>
inline __device__ void block_256_reduce_min(T * cache_T, int tid);

// Warp Reduce maximum
template<typename T>
inline __device__ void warp_reduce_max(volatile T * cache_T, int tid);
// Block of 256 threads Reduce maximum
template<typename T>
inline __device__ void block_256_reduce_max(T * cache_T, int tid);





// HashEntry * entry
// My_Type::Vector3i * order_pos
//! Build allocate flag for allcation stage. (CUDA kernel function)
__global__ void build_entry_flag_KernelFunc(const My_Type::Vector3f * current_points, 
											const float * map_pose, 
											const HashEntry * entry, 
											char * allocate_flag, 
											My_Type::Vector3i * order_pos)
{
	// 
	int u, v, image_width, index;
	u = threadIdx.x + blockIdx.x * blockDim.x;
	v = threadIdx.y + blockIdx.y * blockDim.y;
	image_width = blockDim.x * gridDim.x;
	index = u + v * image_width;
	// 
	My_Type::Vector3f camera_point = current_points[index];
	// 
	if (camera_point.z <= FLT_EPSILON)		return;
	
	// 
	My_Type::Vector3f camera_ray = camera_point;
	// Normalize
	camera_ray /= norm3df(camera_ray.x, camera_ray.y, camera_ray.z);

	My_Type::Vector3f world_point, ray_direction;
	// Convert to world coordinate
	world_point.x = map_pose[0] * camera_point.x + map_pose[4] * camera_point.y + map_pose[8] * camera_point.z + map_pose[12];
	world_point.y = map_pose[1] * camera_point.x + map_pose[5] * camera_point.y + map_pose[9] * camera_point.z + map_pose[13];
	world_point.z = map_pose[2] * camera_point.x + map_pose[6] * camera_point.y + map_pose[10] * camera_point.z + map_pose[14];
	// Compute ray direction
	ray_direction.x = map_pose[0] * camera_ray.x + map_pose[4] * camera_ray.y + map_pose[8] * camera_ray.z;
	ray_direction.y = map_pose[1] * camera_ray.x + map_pose[5] * camera_ray.y + map_pose[9] * camera_ray.z;
	ray_direction.z = map_pose[2] * camera_ray.x + map_pose[6] * camera_ray.y + map_pose[10] * camera_ray.z;

	// direction flag
	My_Type::Vector3i direction_flag;
	direction_flag.x = (ray_direction.x > 0.0 ? 1 : -1);
	direction_flag.y = (ray_direction.y > 0.0 ? 1 : -1);
	direction_flag.z = (ray_direction.z > 0.0 ? 1 : -1);

	// 
	My_Type::Vector3f ray_start;
	ray_start = world_point - ray_direction * TRUNCATED_BAND;
	// 
	My_Type::Vector3i ray_step_line, block_position;

	// 
	My_Type::Vector3f ray_prediction_t;
	float ray_min_t;
	// 
	float ray_current_t;
	// 
	My_Type::Vector3f ray_current;


	// 
	ray_current.x = ray_start.x * 1000.0f;
	ray_current.y = ray_start.y * 1000.0f;
	ray_current.z = ray_start.z * 1000.0f;
	ray_step_line.x = (int)floorf(ray_current.x / DDA_STEP_SIZE_MM);
	ray_step_line.y = (int)floorf(ray_current.y / DDA_STEP_SIZE_MM);
	ray_step_line.z = (int)floorf(ray_current.z / DDA_STEP_SIZE_MM);
	if (ray_direction.x > 0.0)	ray_step_line.x++;
	if (ray_direction.y > 0.0)	ray_step_line.y++;
	if (ray_direction.z > 0.0)	ray_step_line.z++;


	// 
	ray_current_t = 0;
	while (ray_current_t < 2 * TRUNCATED_BAND_MM)
	{
#pragma region(Mark entries)

		//
		block_position.x = ray_step_line.x;
		block_position.y = ray_step_line.y;
		block_position.z = ray_step_line.z;
		if (direction_flag.x > 0)	block_position.x--;
		if (direction_flag.y > 0)	block_position.y--;
		if (direction_flag.z > 0)	block_position.z--;

		// 
		int hash_value = hash_func(block_position.x, block_position.y, block_position.z);

		// 
		bool is_allocated = false;
		if ((entry[hash_value].position[0] == block_position.x) && \
			(entry[hash_value].position[1] == block_position.y) && \
			(entry[hash_value].position[2] == block_position.z))
		{
			is_allocated = true;
		}


		// empty entry in ordered_entry 
		if (!is_allocated)
		{
			// encounter hash collision
			int excess_offset = entry[hash_value].offset;
			// find in excess_entry
			while (excess_offset >= 0)
			{
				int pre_offset = excess_offset;

				// 
				if ((entry[pre_offset].position[0] == block_position.x) && \
					(entry[pre_offset].position[1] == block_position.y) && \
					(entry[pre_offset].position[2] == block_position.z))
				{
					is_allocated = true;
					break;
				}

				// 
				excess_offset = entry[pre_offset].offset;
			}
		}

		// 
		if (!is_allocated)
		{
			// 
			allocate_flag[hash_value] = NEED_ALLOCATE;
			//  
			order_pos[hash_value].x = block_position.x;
			order_pos[hash_value].y = block_position.y;
			order_pos[hash_value].z = block_position.z;

		}

#pragma endregion


		// 
		ray_prediction_t.x = ((float)ray_step_line.x * DDA_STEP_SIZE_MM - ray_current.x) / ray_direction.x;
		ray_prediction_t.y = ((float)ray_step_line.y * DDA_STEP_SIZE_MM - ray_current.y) / ray_direction.y;
		ray_prediction_t.z = ((float)ray_step_line.z * DDA_STEP_SIZE_MM - ray_current.z) / ray_direction.z;
		ray_min_t = ray_prediction_t.x;
		int min_direction = 0;
		if (ray_min_t > ray_prediction_t.y)
		{
			ray_min_t = ray_prediction_t.y;
			min_direction = 1;
		}
		if (ray_min_t > ray_prediction_t.z)
		{
			ray_min_t = ray_prediction_t.z;
			min_direction = 2;
		}

		// 
		ray_step_line[min_direction] += direction_flag[min_direction];


		// 
		ray_current.x += ray_min_t * ray_direction.x;
		ray_current.y += ray_min_t * ray_direction.y;
		ray_current.z += ray_min_t * ray_direction.z;
		ray_current_t += ray_min_t;
	}
}			
//! Build allocate flag of entries by current frame points. (Call CUDA kernel function)
/*!
	\param	block_rect	

	\param	thread_rect	

	\return	void
*/
void build_entry_flag_CUDA(dim3 block_rect, dim3 thread_rect, 
						   const My_Type::Vector3f * current_points, 
						   const float * map_pose, 
						   const HashEntry * entry, 
						   char * allocate_flag, 
						   My_Type::Vector3i * order_pos)
{

	build_entry_flag_KernelFunc << <block_rect, thread_rect >> >(current_points, map_pose, entry, allocate_flag, order_pos);
}


#pragma region(Build allocate flag by another voxel map)
// Offset array
__device__ __constant__ int offset_array[81] = {
	-1, -1, -1,
	0, -1, -1,
	1, -1, -1,
	-1, 0, -1,
	0, 0, -1,
	1, 0, -1,
	-1, 1, -1,
	0, 1, -1,
	1, 1, -1,
	-1, -1, 0,
	0, -1, 0,
	1, -1, 0,
	-1, 0, 0,
	0, 0, 0,
	1, 0, 0,
	-1, 1, 0,
	0, 1, 0,
	1, 1, 0,
	-1, -1, 1,
	0, -1, 1,
	1, -1, 1,
	-1, 0, 1,
	0, 0, 1,
	1, 0, 1,
	-1, 1, 1,
	0, 1, 1,
	1, 1, 1, };

// 
__global__ void build_entry_flag_KernelFunc(My_Type::Vector3i * block_position, int block_num, float * pose, \
											HashEntry * entry, char * allocate_flag, My_Type::Vector3i * order_pos)
{
	// block position array 
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= block_num)	return;

	// 
	My_Type::Vector3f position, block_center;
	position.x = (0.5 + (float)block_position[index].x) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	position.y = (0.5 + (float)block_position[index].y) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	position.z = (0.5 + (float)block_position[index].z) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;

	block_center.x = pose[0] * position.x + pose[4] * position.y + pose[8] * position.z + pose[12];
	block_center.y = pose[1] * position.x + pose[5] * position.y + pose[9] * position.z + pose[13];
	block_center.z = pose[2] * position.x + pose[6] * position.y + pose[10] * position.z + pose[14];
	//
	My_Type::Vector3i block_base_offset;
	block_base_offset.x = (int)(roundf(block_center.x / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));
	block_base_offset.y = (int)(roundf(block_center.y / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));
	block_base_offset.z = (int)(roundf(block_center.z / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));


	//
	for (int i = 0; i < 27; i++)
	{
		//
		My_Type::Vector3i block_offset = block_base_offset;
		block_offset.x += offset_array[3 * i + 0];
		block_offset.y += offset_array[3 * i + 1];
		block_offset.z += offset_array[3 * i + 2];


		int hash_value = hash_func(block_offset.x, block_offset.y, block_offset.z);

		bool is_allocated = false;
		if ((entry[hash_value].position[0] == block_offset.x) && \
			(entry[hash_value].position[1] == block_offset.y) && \
			(entry[hash_value].position[2] == block_offset.z))
		{
			is_allocated = true;
		}


		if (!is_allocated)
		{
			int excess_offset = entry[hash_value].offset;
			if (excess_offset >= 0)
			{
				while (excess_offset >= 0)
				{
					int pre_offset = excess_offset;

					if ((entry[pre_offset].position[0] == block_offset.x) && \
						(entry[pre_offset].position[1] == block_offset.y) && \
						(entry[pre_offset].position[2] == block_offset.z))
					{
						is_allocated = true;
						break;
					}

					excess_offset = entry[pre_offset].offset;
				}
			}

		}

		if (!is_allocated)
		{
			allocate_flag[hash_value] = NEED_ALLOCATE;
			order_pos[hash_value].x = block_offset.x;
			order_pos[hash_value].y = block_offset.y;
			order_pos[hash_value].z = block_offset.z;
		}
	}


}
// Cpp call CUDA
void build_entry_flag_CUDA(dim3 block_rect, dim3 thread_rect, My_Type::Vector3i * block_position, int block_num, float * pose, \
						   HashEntry * entry, char * allocate_flag, My_Type::Vector3i * order_pos)
{

	build_entry_flag_KernelFunc << <block_rect, thread_rect >> >(block_position, block_num, pose, entry, allocate_flag, order_pos);
}

#pragma endregion


//! Allocate voxel blocks. (link entry to block)
__global__ void allocate_by_flag_KernelFunc(HashEntry * entry, 
											const char * allocate_flag, 
											My_Type::Vector3i * pos_buffer, 
											int * excess_counter, 
											int * number_of_blocks, 
											My_Type::Vector3i * voxel_block_position)
{
	int hash_value = threadIdx.x + blockIdx.x * blockDim.x, excess_offset, pre_offset;
	if (allocate_flag[hash_value] == NOT_NEED_ALLOCATE)	return;

	My_Type::Vector3i block_position = pos_buffer[hash_value];
	bool is_collision = false;
	if (entry[hash_value].ptr >= 0)
	{
		is_collision = true;

		bool linked_to_excess = false;
		excess_offset = entry[hash_value].offset;
		if (excess_offset >= 0)
		{
			linked_to_excess = true;
			while (excess_offset >= 0)
			{
				pre_offset = excess_offset;
				excess_offset = entry[pre_offset].offset;
			}

			excess_offset = atomicAdd(excess_counter, 1) + ORDERED_TABLE_LENGTH;
			entry[pre_offset].offset = excess_offset;
		}


		if (!linked_to_excess)
		{
			excess_offset = atomicAdd(excess_counter, 1) + ORDERED_TABLE_LENGTH;
			entry[hash_value].offset = excess_offset;
		}

		// ------------------------ To Do : out put error state
		if (excess_offset > EXCESS_TABLE_LENGTH + ORDERED_TABLE_LENGTH)	return;
		
		entry[excess_offset].position[0] = block_position.x;
		entry[excess_offset].position[1] = block_position.y;
		entry[excess_offset].position[2] = block_position.z;
		int index = atomicAdd(number_of_blocks, 1);
		entry[excess_offset].ptr = index * (int)VOXEL_BLOCK_SIZE;
		voxel_block_position[index] = block_position;
	}

	if (!is_collision)
	{
		entry[hash_value].position[0] = block_position.x;
		entry[hash_value].position[1] = block_position.y;
		entry[hash_value].position[2] = block_position.z;
		int index = atomicAdd(number_of_blocks, 1);
		entry[hash_value].ptr = index * (int)VOXEL_BLOCK_SIZE;
		voxel_block_position[index] = block_position;
	}

}
// Cpp call CUDA
void allocate_by_flag_CUDA(dim3 block_rect, dim3 thread_rect, 
						   HashEntry * entry, 
						   const char * allocate_flag, 
						   My_Type::Vector3i * pos_buffer, 
						   int * excess_counter, 
						   int * number_of_blocks, 
						   My_Type::Vector3i * voxel_block_position)
{
	allocate_by_flag_KernelFunc << <block_rect, thread_rect >> >(entry, allocate_flag, pos_buffer, excess_counter, number_of_blocks, voxel_block_position);
}



__global__ void build_visible_flag_KernelFunc(const My_Type::Vector3f * current_points, 
											  const float * camera_pose, 
											  HashEntry * entry,
											  char * visible_flag)
{
	int u, v, image_width, index;
	u = threadIdx.x + blockIdx.x * blockDim.x;
	v = threadIdx.y + blockIdx.y * blockDim.y;
	image_width = blockDim.x * gridDim.x;
	index = u + v * image_width;
	My_Type::Vector3f camera_point = current_points[index];
	if (camera_point.z <= FLT_EPSILON)		return;

	My_Type::Vector3f camera_ray = camera_point;
	camera_ray /= norm3df(camera_ray.x, camera_ray.y, camera_ray.z);

	My_Type::Vector3f world_point, ray_direction;
	world_point.x = camera_pose[0] * camera_point.x + camera_pose[4] * camera_point.y + camera_pose[8] * camera_point.z + camera_pose[12];
	world_point.y = camera_pose[1] * camera_point.x + camera_pose[5] * camera_point.y + camera_pose[9] * camera_point.z + camera_pose[13];
	world_point.z = camera_pose[2] * camera_point.x + camera_pose[6] * camera_point.y + camera_pose[10] * camera_point.z + camera_pose[14];
	ray_direction.x = camera_pose[0] * camera_ray.x + camera_pose[4] * camera_ray.y + camera_pose[8] * camera_ray.z;
	ray_direction.y = camera_pose[1] * camera_ray.x + camera_pose[5] * camera_ray.y + camera_pose[9] * camera_ray.z;
	ray_direction.z = camera_pose[2] * camera_ray.x + camera_pose[6] * camera_ray.y + camera_pose[10] * camera_ray.z;

	My_Type::Vector3i direction_flag;
	direction_flag.x = (ray_direction.x > 0.0 ? 1 : -1);
	direction_flag.y = (ray_direction.y > 0.0 ? 1 : -1);
	direction_flag.z = (ray_direction.z > 0.0 ? 1 : -1);

	My_Type::Vector3f ray_start;
	ray_start = world_point - ray_direction * TRUNCATED_BAND;
	// 
	My_Type::Vector3i ray_step_line, block_position;

	My_Type::Vector3f ray_prediction_t;
	float ray_min_t;
	float ray_current_t;
	My_Type::Vector3f ray_current;

	ray_current.x = ray_start.x * 1000.0f;
	ray_current.y = ray_start.y * 1000.0f;
	ray_current.z = ray_start.z * 1000.0f;
	ray_step_line.x = (int)floorf(ray_current.x / DDA_STEP_SIZE_MM);
	ray_step_line.y = (int)floorf(ray_current.y / DDA_STEP_SIZE_MM);
	ray_step_line.z = (int)floorf(ray_current.z / DDA_STEP_SIZE_MM);
	if (ray_direction.x > 0.0)	ray_step_line.x++;
	if (ray_direction.y > 0.0)	ray_step_line.y++;
	if (ray_direction.z > 0.0)	ray_step_line.z++;


	ray_current_t = 0;
	while (ray_current_t < 2 * TRUNCATED_BAND_MM)
	{
#pragma region(Mark entries)

		block_position = ray_step_line;
		if (direction_flag.x > 0)	block_position.x--;
		if (direction_flag.y > 0)	block_position.y--;
		if (direction_flag.z > 0)	block_position.z--;

		int hash_value = hash_func(block_position.x, block_position.y, block_position.z);

		bool is_allocated = false;
		if ((entry[hash_value].position[0] == block_position.x) && \
			(entry[hash_value].position[1] == block_position.y) && \
			(entry[hash_value].position[2] == block_position.z))
		{
			is_allocated = true;

			visible_flag[hash_value] = VISIBLE_BLOCK;
		}


		if (!is_allocated)
		{
			int excess_offset = entry[hash_value].offset;
			if (excess_offset >= 0)
			{
				while (excess_offset >= 0)
				{
					int pre_offset = excess_offset;

					if ((entry[pre_offset].position[0] == block_position.x) && \
						(entry[pre_offset].position[1] == block_position.y) && \
						(entry[pre_offset].position[2] == block_position.z))
					{
						is_allocated = true;
						visible_flag[excess_offset] = VISIBLE_BLOCK;
						break;
					}

					excess_offset = entry[pre_offset].offset;
				}
			}

		}


#pragma endregion


		ray_prediction_t.x = ((float)ray_step_line.x * DDA_STEP_SIZE_MM - ray_current.x) / ray_direction.x;
		ray_prediction_t.y = ((float)ray_step_line.y * DDA_STEP_SIZE_MM - ray_current.y) / ray_direction.y;
		ray_prediction_t.z = ((float)ray_step_line.z * DDA_STEP_SIZE_MM - ray_current.z) / ray_direction.z;
		ray_min_t = ray_prediction_t.x;
		int min_direction = 0;
		if (ray_min_t > ray_prediction_t.y)
		{
			ray_min_t = ray_prediction_t.y;
			min_direction = 1;
		}
		if (ray_min_t > ray_prediction_t.z)
		{
			ray_min_t = ray_prediction_t.z;
			min_direction = 2;
		}

		ray_step_line[min_direction] += direction_flag[min_direction];


		ray_current.x += ray_min_t * ray_direction.x;
		ray_current.y += ray_min_t * ray_direction.y;
		ray_current.z += ray_min_t * ray_direction.z;
		ray_current_t += ray_min_t;
	}
}
// Cpp call CUDA
void build_visible_flag_CUDA(dim3 block_rect, dim3 thread_rect, 
							 const My_Type::Vector3f * current_points, 
							 const float * camera_pose, 
							 HashEntry * entry, 
							 char * visible_flag)
{
	build_visible_flag_KernelFunc << <block_rect, thread_rect >> >(current_points, camera_pose, entry, visible_flag);
}


__global__ void build_relative_flag_KernelFunc(My_Type::Vector3i * block_position, int block_num, float * pose, \
	HashEntry * entry, char * relative_flag)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= block_num)	return;

	My_Type::Vector3f position, block_center;
	position.x = (0.5 + (float)block_position[index].x) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	position.y = (0.5 + (float)block_position[index].y) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	position.z = (0.5 + (float)block_position[index].z) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;

	block_center.x = pose[0] * position.x + pose[4] * position.y + pose[8] * position.z + pose[12];
	block_center.y = pose[1] * position.x + pose[5] * position.y + pose[9] * position.z + pose[13];
	block_center.z = pose[2] * position.x + pose[6] * position.y + pose[10] * position.z + pose[14];
	//
	My_Type::Vector3i block_base_offset;
	block_base_offset.x = (int)(roundf(block_center.x / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));
	block_base_offset.y = (int)(roundf(block_center.y / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));
	block_base_offset.z = (int)(roundf(block_center.z / (VOXEL_BLOCK_WDITH * VOXEL_SIZE)));

	//
	for (int i = 0; i < 27; i++)
	{
		//
		My_Type::Vector3i block_offset = block_base_offset;
		block_offset.x += offset_array[3 * i + 0];
		block_offset.y += offset_array[3 * i + 1];
		block_offset.z += offset_array[3 * i + 2];


		int hash_value = hash_func(block_offset.x, block_offset.y, block_offset.z);

		bool is_allocated = false;
		if ((entry[hash_value].position[0] == block_offset.x) && \
			(entry[hash_value].position[1] == block_offset.y) && \
			(entry[hash_value].position[2] == block_offset.z))
		{
			is_allocated = true;
			relative_flag[hash_value] = VISIBLE_BLOCK;
		}


		if (!is_allocated)
		{
			int excess_offset = entry[hash_value].offset;
			if (excess_offset >= 0)
			{
				while (excess_offset >= 0)
				{
					int pre_offset = excess_offset;

					if ((entry[pre_offset].position[0] == block_offset.x) && \
						(entry[pre_offset].position[1] == block_offset.y) && \
						(entry[pre_offset].position[2] == block_offset.z))
					{
						is_allocated = true;
						relative_flag[excess_offset] = VISIBLE_BLOCK;
						break;
					}

					excess_offset = entry[pre_offset].offset;
				}
			}

		}
	}


}
// Cpp call CUDA
void build_relative_flag_CUDA(dim3 block_rect, dim3 thread_rect, My_Type::Vector3i * block_position, int block_num, float * pose, \
	HashEntry * entry, char * relative_flag)
{

	build_relative_flag_KernelFunc << <block_rect, thread_rect >> >(block_position, block_num, pose, entry, relative_flag);
}



__global__ void build_visible_list_KernelFunc(const HashEntry * entry, 
											  HashEntry * visible_entries, 
											  const char * visible_flag, 
											  int * visible_counter, 
											  const float * pose_inv, 
											  int * min_depth, int * max_depth)
{
	int hash_value = threadIdx.x + blockIdx.x * blockDim.x;

	if (visible_flag[hash_value] == INVISIBLE_BLOCK)	return;

	int index = atomicAdd(visible_counter, 1);
	HashEntry entry_cache = entry[hash_value];
	visible_entries[index] = entry_cache;

	float current_depth;
	current_depth = (pose_inv[2] * entry_cache.position[0] + pose_inv[6] * entry_cache.position[1] + pose_inv[10] * entry_cache.position[2]) * DDA_STEP_SIZE_MM 
					+ pose_inv[14] * 1000.0f;
	//
	current_depth -= DDA_STEP_SIZE_MM;
	atomicMin(min_depth, (int)current_depth);
	current_depth += DDA_STEP_SIZE_MM;
	atomicMax(max_depth, (int)current_depth);

}
// Cpp call CUDA 
void build_visible_list_CUDA(dim3 block_rect, dim3 thread_rect, 
							 const HashEntry * entry, 
							 HashEntry * visible_list, 
							 const char * visible_flag,
							 int * visible_counter, 
							 const float * pose_inv, 
							 int * min_depth, int * max_depth)
{
	build_visible_list_KernelFunc << <block_rect, thread_rect >> >(entry, visible_list, visible_flag, visible_counter, pose_inv, min_depth, max_depth);
}


//
__global__ void build_relative_list_KernelFunc(HashEntry * entry, HashEntry * relative_list, char * relative_flag, int * relative_counter, float * pose_inv)
{
	int hash_value = threadIdx.x + blockIdx.x * blockDim.x;

	if (relative_flag[hash_value] == 0)	return;


	int index = atomicAdd(relative_counter, 1);
	HashEntry entry_cache = entry[hash_value];
	relative_list[index] = entry_cache;
}
// Cpp call CUDA
void build_relative_list_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * entry, HashEntry * relative_list, char * relative_flag, int * relative_counter, float * pose_inv)
{

	build_relative_list_KernelFunc << <block_rect, thread_rect >> >(entry, relative_list, relative_flag, relative_counter, pose_inv);
}



__global__ void raycast_get_range_KernelFunc(const float * view_pose, 
											 const HashEntry * entry,
											 Sensor_params sensor_params,
											 int raycast_patch_width,
											 const int * min_distance, const int * max_distance, 
											 My_Type::Vector2f * range_map)
{
	float camera_start_x, camera_start_y, camera_start_z;
	float camera_direction_x, camera_direction_y, camera_direction_z, direction_length;
	float direction_x, direction_y, direction_z;
	int direction_flag[3];

	int u, v, u_index, v_index;
	u_index = threadIdx.x;
	v_index = blockIdx.x;
	u = u_index * raycast_patch_width + raycast_patch_width / 2;
	v = v_index * raycast_patch_width + raycast_patch_width / 2;

	//
	float min_depth, max_depth;
	min_depth = 0.001f * (float)min_distance[0] - DDA_STEP_SIZE;
	max_depth = 0.001f * (float)max_distance[0] + DDA_STEP_SIZE;
	if (min_depth >= max_depth)		return;
	camera_start_x = min_depth * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_start_y = min_depth * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_start_z = min_depth;
	camera_direction_x = max_depth * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_direction_y = max_depth * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_direction_z = max_depth;
	camera_direction_x -= camera_start_x;
	camera_direction_y -= camera_start_y;
	camera_direction_z -= camera_start_z;
	direction_length = norm3df(camera_direction_x, camera_direction_y, camera_direction_z);
	camera_direction_x /= direction_length;	camera_direction_y /= direction_length;	camera_direction_z /= direction_length;


	float point_x, point_y, point_z;
	point_x = view_pose[0] * camera_start_x + view_pose[4] * camera_start_y + view_pose[8] * camera_start_z + view_pose[12];
	point_y = view_pose[1] * camera_start_x + view_pose[5] * camera_start_y + view_pose[9] * camera_start_z + view_pose[13];
	point_z = view_pose[2] * camera_start_x + view_pose[6] * camera_start_y + view_pose[10] * camera_start_z + view_pose[14];
	direction_x = view_pose[0] * camera_direction_x + view_pose[4] * camera_direction_y + view_pose[8] * camera_direction_z;
	direction_y = view_pose[1] * camera_direction_x + view_pose[5] * camera_direction_y + view_pose[9] * camera_direction_z;
	direction_z = view_pose[2] * camera_direction_x + view_pose[6] * camera_direction_y + view_pose[10] * camera_direction_z;

	direction_flag[0] = (direction_x > 0.0 ? 1 : -1);
	direction_flag[1] = (direction_y > 0.0 ? 1 : -1);
	direction_flag[2] = (direction_z > 0.0 ? 1 : -1);


	float current[3];
	int step_line[3];
	int block_pos[3];
	float t_coeff[3], min_coeff, max_t_coeff, current_t_coeff;
	int min_direction;
	current_t_coeff = 0.0;
	max_t_coeff = direction_length;
	int hash_value, excess_offset;

	current[0] = point_x;	current[1] = point_y;	current[2] = point_z;
	step_line[0] = (int)floorf(current[0] / DDA_STEP_SIZE);
	step_line[1] = (int)floorf(current[1] / DDA_STEP_SIZE);
	step_line[2] = (int)floorf(current[2] / DDA_STEP_SIZE);
	if (direction_x > 0.0)	step_line[0] ++;
	if (direction_y > 0.0)	step_line[1] ++;
	if (direction_z > 0.0)	step_line[2] ++;


	//
	bool found_allocated_before = false;
	bool is_find_near = false;
	while (current_t_coeff < max_t_coeff)
	{

#pragma region(Get Range)

		// 
		block_pos[0] = step_line[0];
		block_pos[1] = step_line[1];
		block_pos[2] = step_line[2];
		if (direction_flag[0] > 0)	block_pos[0]--;
		if (direction_flag[1] > 0)	block_pos[1]--;
		if (direction_flag[2] > 0)	block_pos[2]--;

		// 
		hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);

		bool is_allocated = false;
		if ((entry[hash_value].position[0] == block_pos[0]) && \
			(entry[hash_value].position[1] == block_pos[1]) && \
			(entry[hash_value].position[2] == block_pos[2]))
		{
			is_allocated = true;
		}

		if (!is_allocated)
		{
			excess_offset = entry[hash_value].offset;
			if (excess_offset >= 0)
			{
				while (excess_offset >= 0)
				{
					if ((entry[excess_offset].position[0] == block_pos[0]) && \
						(entry[excess_offset].position[1] == block_pos[1]) && \
						(entry[excess_offset].position[2] == block_pos[2]))
					{
						is_allocated = true;
						break;
					}

					excess_offset = entry[excess_offset].offset;
				}
			}


		}


		if (is_allocated)
		{
			//
			found_allocated_before = true;

			if (!is_find_near)
			{
				is_find_near = true;
				//
				//range_map[u_index + v_index * blockDim.x].x = min_depth + camera_direction_z * current_t_coeff - 2 * DDA_STEP_SIZE;
				int range_map_width = blockDim.x + 1;
				range_map[u_index + v_index * range_map_width].x = min_depth + camera_direction_z * current_t_coeff;

			}
		}
		if (!is_allocated)
		{
			if (found_allocated_before)
			{
				found_allocated_before = false;
				int range_map_width = blockDim.x + 1;
				//range_map[u_index + v_index * range_map_width].y = min_depth + camera_direction_z * current_t_coeff + 2 * DDA_STEP_SIZE;
				range_map[u_index + v_index * range_map_width].y = min_depth + camera_direction_z * current_t_coeff;
			}
		}


#pragma endregion


		step_line[0] = (int)floorf(current[0] / DDA_STEP_SIZE);
		step_line[1] = (int)floorf(current[1] / DDA_STEP_SIZE);
		step_line[2] = (int)floorf(current[2] / DDA_STEP_SIZE);
		if (direction_x > 0.0)	step_line[0] ++;
		if (direction_y > 0.0)	step_line[1] ++;
		if (direction_z > 0.0)	step_line[2] ++;
		//
		if (fabsf((float)step_line[0] * DDA_STEP_SIZE - current[0]) < FLT_EPSILON)	step_line[0] += direction_flag[0];
		if (fabsf((float)step_line[1] * DDA_STEP_SIZE - current[1]) < FLT_EPSILON)	step_line[1] += direction_flag[1];
		if (fabsf((float)step_line[2] * DDA_STEP_SIZE - current[2]) < FLT_EPSILON)	step_line[2] += direction_flag[2];


		t_coeff[0] = ((float)step_line[0] * DDA_STEP_SIZE - current[0]) / direction_x;
		t_coeff[1] = ((float)step_line[1] * DDA_STEP_SIZE - current[1]) / direction_y;
		t_coeff[2] = ((float)step_line[2] * DDA_STEP_SIZE - current[2]) / direction_z;
		min_coeff = t_coeff[0];			min_direction = 0;
		if (min_coeff > t_coeff[1])
		{
			min_coeff = t_coeff[1];
			min_direction = 1;
		}
		if (min_coeff > t_coeff[2])
		{
			min_coeff = t_coeff[2];
			min_direction = 2;
		}
		min_coeff = max(HALF_VOXEL_SIZE, min_coeff);

		step_line[min_direction] += direction_flag[min_direction];

		current[0] += min_coeff * direction_x;
		current[1] += min_coeff * direction_y;
		current[2] += min_coeff * direction_z;
		current_t_coeff += min_coeff;

	}

}
// Cpp call CUDA 
void raycast_get_range_CUDA(dim3 block_rect, dim3 thread_rect, 
							const float * view_pose, 
							const HashEntry * entry,
							Sensor_params sensor_params,
							int raycast_patch_width,
							const int * min_distance, const int * max_distance,
							My_Type::Vector2f * range_map)
{
	raycast_get_range_KernelFunc << <block_rect, thread_rect >> >(view_pose, entry, sensor_params, raycast_patch_width, min_distance, max_distance, range_map);
}


__global__ void raycast_get_range_4corner_KernelFunc(const float * view_pose,
													 Sensor_params sensor_params,
													 int raycast_patch_width,
													 const HashEntry * entry,
													 const int * min_distance, const int * max_distance, 
													 My_Type::Vector2f * range_map)
{
	float camera_start_x, camera_start_y, camera_start_z;
	float camera_direction_x, camera_direction_y, camera_direction_z, direction_length;
	float direction_x, direction_y, direction_z;
	int direction_flag[3];
	int patch_corner[4][2] = { 	
		{ 0,					0}, 
		{ 0,					raycast_patch_width }, 	
		{ raycast_patch_width,	0			}, 	
		{ raycast_patch_width,	raycast_patch_width }	
	};
	// 
	float patch_min, patch_max;
	//patch_min = max_distance[0];
	//patch_max = min_distance[0];
	patch_min = +FLT_MAX;
	patch_max = -FLT_MAX;

	//
	float min_depth, max_depth;
	min_depth = 0.001f * (float)min_distance[0] - DDA_STEP_SIZE;
	max_depth = 0.001f * (float)max_distance[0] + DDA_STEP_SIZE;
	if (min_depth >= max_depth)		return;


	int u, v, u_index, v_index;
	u_index = threadIdx.x;
	v_index = blockIdx.x;


	for (int i = 0; i < 4; i++)
	{
		u = u_index * raycast_patch_width + patch_corner[i][0];
		v = v_index * raycast_patch_width + patch_corner[i][1];

		camera_start_x = min_depth * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
		camera_start_y = min_depth * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
		camera_start_z = min_depth;
		camera_direction_x = max_depth * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
		camera_direction_y = max_depth * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
		camera_direction_z = max_depth;
		camera_direction_x -= camera_start_x;
		camera_direction_y -= camera_start_y;
		camera_direction_z -= camera_start_z;
		direction_length = norm3df(camera_direction_x, camera_direction_y, camera_direction_z);
		camera_direction_x /= direction_length;	camera_direction_y /= direction_length;	camera_direction_z /= direction_length;


		float point_x, point_y, point_z;
		point_x = view_pose[0] * camera_start_x + view_pose[4] * camera_start_y + view_pose[8] * camera_start_z + view_pose[12];
		point_y = view_pose[1] * camera_start_x + view_pose[5] * camera_start_y + view_pose[9] * camera_start_z + view_pose[13];
		point_z = view_pose[2] * camera_start_x + view_pose[6] * camera_start_y + view_pose[10] * camera_start_z + view_pose[14];
		direction_x = view_pose[0] * camera_direction_x + view_pose[4] * camera_direction_y + view_pose[8] * camera_direction_z;
		direction_y = view_pose[1] * camera_direction_x + view_pose[5] * camera_direction_y + view_pose[9] * camera_direction_z;
		direction_z = view_pose[2] * camera_direction_x + view_pose[6] * camera_direction_y + view_pose[10] * camera_direction_z;

		direction_flag[0] = (direction_x > 0.0 ? 1 : -1);
		direction_flag[1] = (direction_y > 0.0 ? 1 : -1);
		direction_flag[2] = (direction_z > 0.0 ? 1 : -1);


		float current[3];
		int step_line[3];
		int block_pos[3];
		float t_coeff[3], min_coeff, max_t_coeff, current_t_coeff;
		int min_direction;
		current_t_coeff = 0.0;
		max_t_coeff = direction_length;
		int hash_value, excess_offset;

		current[0] = point_x;	current[1] = point_y;	current[2] = point_z;
		step_line[0] = (int)floorf(current[0] / DDA_STEP_SIZE);
		step_line[1] = (int)floorf(current[1] / DDA_STEP_SIZE);
		step_line[2] = (int)floorf(current[2] / DDA_STEP_SIZE);
		if (direction_x > 0.0)	step_line[0] ++;
		if (direction_y > 0.0)	step_line[1] ++;
		if (direction_z > 0.0)	step_line[2] ++;


		//
		bool found_allocated_before = false;
		bool is_find_near = false;
		while (current_t_coeff < max_t_coeff)
		{

#pragma region(Get range)

			block_pos[0] = step_line[0];
			block_pos[1] = step_line[1];
			block_pos[2] = step_line[2];
			if (direction_flag[0] > 0)	block_pos[0]--;
			if (direction_flag[1] > 0)	block_pos[1]--;
			if (direction_flag[2] > 0)	block_pos[2]--;

			hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);

			bool is_allocated = false;
			if ((entry[hash_value].position[0] == block_pos[0]) && \
				(entry[hash_value].position[1] == block_pos[1]) && \
				(entry[hash_value].position[2] == block_pos[2]))
			{
				is_allocated = true;
			}

			if (!is_allocated)
			{
				excess_offset = entry[hash_value].offset;
				if (excess_offset >= 0)
				{
					while (excess_offset >= 0)
					{
						if ((entry[excess_offset].position[0] == block_pos[0]) && \
							(entry[excess_offset].position[1] == block_pos[1]) && \
							(entry[excess_offset].position[2] == block_pos[2]))
						{
							is_allocated = true;
							break;
						}

						excess_offset = entry[excess_offset].offset;
					}
				}


			}


			if (is_allocated)
			{
				//
				found_allocated_before = true;

				if (!is_find_near)
				{
					is_find_near = true;
					patch_min = min(patch_min, min_depth + camera_direction_z * current_t_coeff);
				}
			}
			if (!is_allocated)
			{
				if (found_allocated_before)
				{
					found_allocated_before = false;
					patch_max = max(patch_max, min_depth + camera_direction_z * current_t_coeff);
				}
			}


#pragma endregion


			step_line[0] = (int)floorf(current[0] / DDA_STEP_SIZE);
			step_line[1] = (int)floorf(current[1] / DDA_STEP_SIZE);
			step_line[2] = (int)floorf(current[2] / DDA_STEP_SIZE);
			if (direction_x > 0.0)	step_line[0] ++;
			if (direction_y > 0.0)	step_line[1] ++;
			if (direction_z > 0.0)	step_line[2] ++;
			//
			if (fabsf((float)step_line[0] * DDA_STEP_SIZE - current[0]) < FLT_EPSILON)	step_line[0] += direction_flag[0];
			if (fabsf((float)step_line[1] * DDA_STEP_SIZE - current[1]) < FLT_EPSILON)	step_line[1] += direction_flag[1];
			if (fabsf((float)step_line[2] * DDA_STEP_SIZE - current[2]) < FLT_EPSILON)	step_line[2] += direction_flag[2];


			t_coeff[0] = ((float)step_line[0] * DDA_STEP_SIZE - current[0]) / direction_x;
			t_coeff[1] = ((float)step_line[1] * DDA_STEP_SIZE - current[1]) / direction_y;
			t_coeff[2] = ((float)step_line[2] * DDA_STEP_SIZE - current[2]) / direction_z;
			min_coeff = t_coeff[0];			min_direction = 0;
			if (min_coeff > t_coeff[1])
			{
				min_coeff = t_coeff[1];
				min_direction = 1;
			}
			if (min_coeff > t_coeff[2])
			{
				min_coeff = t_coeff[2];
				min_direction = 2;
			}
			min_coeff = max(HALF_VOXEL_SIZE, min_coeff);

			step_line[min_direction] += direction_flag[min_direction];


			current[0] += min_coeff * direction_x;
			current[1] += min_coeff * direction_y;
			current[2] += min_coeff * direction_z;
			current_t_coeff += min_coeff;
		}
	}


	int range_map_width = blockDim.x + 1;
	range_map[u_index + v_index * range_map_width].x = max(patch_min, 0.0f);
	range_map[u_index + v_index * range_map_width].y = patch_max;
}
// Cpp call CUDA 
void raycast_get_range_4corner_CUDA(dim3 block_rect, dim3 thread_rect,
									const float * view_pose,
									Sensor_params sensor_params,
									int raycast_patch_width,
									const HashEntry * entry,
									const int * min_distance, const int * max_distance,
									My_Type::Vector2f * range_map)
{
	raycast_get_range_4corner_KernelFunc << <block_rect, thread_rect >> >(view_pose, sensor_params, raycast_patch_width, entry, min_distance, max_distance, 
																		  range_map);
}



__global__ void raycast_byStep_KernelFunc(const float * __restrict__ camera_pose,
										  Sensor_params sensor_params,
										  int raycast_patch_width,
										  const HashEntry * __restrict__ entry,
										  const Voxel_f * __restrict__ voxel_block_array,
										  const My_Type::Vector2f * __restrict__ range_map,
										  My_Type::Vector3f * raycast_points,
										  My_Type::Vector3f * raycast_normal,
										  int * raycast_weight)
{
	float camera_start_x, camera_start_y, camera_start_z;
	float camera_direction_x, camera_direction_y, camera_direction_z, direction_length;
	float direction_x, direction_y, direction_z;
	int u, v, u_index, v_index;
	u = threadIdx.x + blockIdx.x * raycast_patch_width;
	v = threadIdx.y + blockIdx.y * raycast_patch_width;
	u_index = blockIdx.x;
	v_index = blockIdx.y;
	float sdf;

	//
	int image_W = blockDim.x * gridDim.x;

	float min_z, max_z;
	int map_W = ceilf((float)image_W / (float)raycast_patch_width);
	min_z = range_map[u_index + map_W * v_index].x;
	max_z = range_map[u_index + map_W * v_index].y;
	if (min_z >= max_z)		return;


	camera_start_x = min_z * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_start_y = min_z * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_start_z = min_z;
	camera_direction_x = max_z * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_direction_y = max_z * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_direction_z = max_z;
	camera_direction_x -= camera_start_x;
	camera_direction_y -= camera_start_y;
	camera_direction_z -= camera_start_z;
	direction_length = 1 / norm3df(camera_direction_x, camera_direction_y, camera_direction_z);
	camera_direction_x *= direction_length;
	camera_direction_y *= direction_length;
	camera_direction_z *= direction_length;


	float point_x, point_y, point_z;
	point_x = camera_pose[0] * camera_start_x + camera_pose[4] * camera_start_y + camera_pose[8] * camera_start_z + camera_pose[12];
	point_y = camera_pose[1] * camera_start_x + camera_pose[5] * camera_start_y + camera_pose[9] * camera_start_z + camera_pose[13];
	point_z = camera_pose[2] * camera_start_x + camera_pose[6] * camera_start_y + camera_pose[10] * camera_start_z + camera_pose[14];
	direction_x = camera_pose[0] * camera_direction_x + camera_pose[4] * camera_direction_y + camera_pose[8] * camera_direction_z;
	direction_y = camera_pose[1] * camera_direction_x + camera_pose[5] * camera_direction_y + camera_pose[9] * camera_direction_z;
	direction_z = camera_pose[2] * camera_direction_x + camera_pose[6] * camera_direction_y + camera_pose[10] * camera_direction_z;
	

	//
	float step_length = TRUNCATED_BAND;
	//
	bool find_small_value = false;
	bool is_allocated = true;
	float ray_length = min_z  / camera_direction_z;
	float max_ray_length = max_z / camera_direction_z;
	while (ray_length < max_ray_length)
	{
		if (is_allocated)
		{
			if (false)
			{
				point_x -= direction_x * step_length;
				point_y -= direction_y * step_length;
				point_z -= direction_z * step_length;
				ray_length -= step_length;
			}

			step_length = TRUNCATED_BAND;

			//int voxel_index = find_voxel_index_round(point_x, point_y, point_z, entry);
			int voxel_index = get_voxel_index_neighbor(point_x, point_y, point_z, entry);
			if (voxel_index == -1)
			{
				point_x += direction_x * step_length;
				point_y += direction_y * step_length;
				point_z += direction_z * step_length;
				ray_length += step_length;
			}

			int weight = voxel_block_array[voxel_index].weight;

			if (weight > MIN_RAYCAST_WEIGHT)
			{
				sdf = voxel_block_array[voxel_index].sdf;


				if (sdf < 0.1f && sdf > -0.5f)
				{
					find_small_value = true;
					break;
				}

				//
				step_length = VOXEL_SIZE;
			}

		}
		else
		{
			step_length = DDA_STEP_SIZE;
		}


		point_x += direction_x * step_length;
		point_y += direction_y * step_length;
		point_z += direction_z * step_length;
		ray_length += step_length;

		is_allocated = check_block_allocated(point_x, point_y, point_z, entry);
	}


	if (find_small_value)
	{
		bool valid_point = get_sdf_interpolated(point_x, point_y, point_z, entry, voxel_block_array, sdf);

		//
		step_length = sdf * TRUNCATED_BAND;
		point_x += direction_x * step_length;
		point_y += direction_y * step_length;
		point_z += direction_z * step_length;
		ray_length += step_length;

		if (valid_point)
		{

			int point_index = u + image_W * v;
			raycast_points[point_index].x = (float)(ray_length * camera_direction_x);
			raycast_points[point_index].y = (float)(ray_length * camera_direction_y);
			raycast_points[point_index].z = (float)(ray_length * camera_direction_z);

			float sdf_x, sdf_y, sdf_z, sdf_gradient;
			float point_x_inc, point_y_inc, point_z_inc;
			point_x_inc = point_x + VOXEL_SIZE;
			point_y_inc = point_y + VOXEL_SIZE;
			point_z_inc = point_z + VOXEL_SIZE;
			
			valid_point &= get_sdf_interpolated(point_x, point_y, point_z, entry, voxel_block_array, sdf);
			valid_point &= get_sdf_interpolated(point_x_inc, point_y, point_z, entry, voxel_block_array, sdf_x);
			valid_point &= get_sdf_interpolated(point_x, point_y_inc, point_z, entry, voxel_block_array, sdf_y);
			valid_point &= get_sdf_interpolated(point_x, point_y, point_z_inc, entry, voxel_block_array, sdf_z);



			if (valid_point)
			{
				sdf_x -= sdf;			sdf_y -= sdf;			sdf_z -= sdf;
				sdf_gradient = norm3df(sdf_x, sdf_y, sdf_z);
				sdf_x /= sdf_gradient;
				sdf_y /= sdf_gradient;
				sdf_z /= sdf_gradient;

				// 
				float norm_camera_x, norm_camera_y, norm_camera_z;
				norm_camera_x = camera_pose[0] * sdf_x + camera_pose[1] * sdf_y + camera_pose[2] * sdf_z;
				norm_camera_y = camera_pose[4] * sdf_x + camera_pose[5] * sdf_y + camera_pose[6] * sdf_z;
				norm_camera_z = camera_pose[8] * sdf_x + camera_pose[9] * sdf_y + camera_pose[10] * sdf_z;

				raycast_normal[point_index].x = norm_camera_x;
				raycast_normal[point_index].y = norm_camera_y;
				raycast_normal[point_index].z = norm_camera_z;


				//int voxel_index = find_voxel_index_round(point_x, point_y, point_z, entry);
				int voxel_index = get_voxel_index_neighbor(point_x, point_y, point_z, entry);
				raycast_weight[point_index] = voxel_block_array[voxel_index].weight;
			}
			else
			{
				raycast_weight[point_index] = 0;
			}
		}


	}




}
// Cpp call CUDA 
void raycast_byStep_CUDA(dim3 block_rect, dim3 thread_rect, 
						 const float * __restrict__ camera_pose,
						 Sensor_params sensor_params,
						 int raycast_patch_width,
						 const HashEntry * __restrict__ entry,
						 const Voxel_f * __restrict__ voxel_block_array,
						 const My_Type::Vector2f * __restrict__ range_map,
						 My_Type::Vector3f * raycast_points,
						 My_Type::Vector3f * raycast_normal,
						 int * raycast_weight)
{
	raycast_byStep_KernelFunc << <block_rect, thread_rect >> >(camera_pose, sensor_params, raycast_patch_width,
															   entry, voxel_block_array, range_map,
															   raycast_points, raycast_normal, raycast_weight);
}
__global__ void raycast_byStep_KernelFunc(const float * __restrict__ camera_pose, 
										  Sensor_params sensor_params,
										  int raycast_patch_width,
										  const HashEntry * __restrict__ entry, 
										  const Voxel_f * __restrict__ voxel_block_array, 
										  const My_Type::Vector2f * __restrict__ range_map, 
										  My_Type::Vector3f * raycast_points, 
										  My_Type::Vector3f * raycast_normal, 
										  int * raycast_plane_label, 
										  int * raycast_weight)
{
	float camera_start_x, camera_start_y, camera_start_z;
	float camera_direction_x, camera_direction_y, camera_direction_z, direction_length;
	float direction_x, direction_y, direction_z;
	int u, v, u_index, v_index;
	u = threadIdx.x + blockIdx.x * raycast_patch_width;
	v = threadIdx.y + blockIdx.y * raycast_patch_width;
	u_index = blockIdx.x;
	v_index = blockIdx.y;
	// SDF 
	float sdf;

	//
	int image_W = blockDim.x * gridDim.x;

	float min_z, max_z;
	// Compute range map width
	int map_W = 1 + (int)ceilf((float)image_W / (float)raycast_patch_width);
	min_z = range_map[u_index + map_W * v_index].x;
	max_z = range_map[u_index + map_W * v_index].y;
	if (min_z >= max_z)		return;


	camera_start_x = min_z * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_start_y = min_z * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_start_z = min_z;
	camera_direction_x = max_z * (float)(u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
	camera_direction_y = max_z * (float)(v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
	camera_direction_z = max_z;
	camera_direction_x -= camera_start_x;
	camera_direction_y -= camera_start_y;
	camera_direction_z -= camera_start_z;
	direction_length = 1 / norm3df(camera_direction_x, camera_direction_y, camera_direction_z);
	camera_direction_x *= direction_length;
	camera_direction_y *= direction_length;
	camera_direction_z *= direction_length;


	float point_x, point_y, point_z;
	point_x = camera_pose[0] * camera_start_x + camera_pose[4] * camera_start_y + camera_pose[8] * camera_start_z + camera_pose[12];
	point_y = camera_pose[1] * camera_start_x + camera_pose[5] * camera_start_y + camera_pose[9] * camera_start_z + camera_pose[13];
	point_z = camera_pose[2] * camera_start_x + camera_pose[6] * camera_start_y + camera_pose[10] * camera_start_z + camera_pose[14];
	direction_x = camera_pose[0] * camera_direction_x + camera_pose[4] * camera_direction_y + camera_pose[8] * camera_direction_z;
	direction_y = camera_pose[1] * camera_direction_x + camera_pose[5] * camera_direction_y + camera_pose[9] * camera_direction_z;
	direction_z = camera_pose[2] * camera_direction_x + camera_pose[6] * camera_direction_y + camera_pose[10] * camera_direction_z;


	//
	float step_length = DDA_STEP_SIZE;
	//
	bool find_small_value = true;
	bool is_allocated = true;
	//       
	float ray_length = min_z / camera_direction_z;
	float max_ray_length = max_z / camera_direction_z;
	
	
	while (ray_length < max_ray_length)
	{
		if (is_allocated)
		{
			//        
			step_length = TRUNCATED_BAND;

			// find round    BUG？！
			//int voxel_index = find_voxel_index_round(point_x, point_y, point_z, entry);
			int voxel_index = get_voxel_index_neighbor(point_x, point_y, point_z, entry);
			//
			if (voxel_index == -1)
			{
				//        
				point_x += direction_x * step_length;
				point_y += direction_y * step_length;
				point_z += direction_z * step_length;
				ray_length += step_length;
				continue;
			}

			//   Voxel    
			int weight = voxel_block_array[voxel_index].weight;

			if (weight > MIN_RAYCAST_WEIGHT)
			{
				sdf = voxel_block_array[voxel_index].sdf;
				//
				if (sdf < 0.3f && sdf > -0.3f)
				{
					find_small_value = true;
					break;
				}

				//
				step_length = VOXEL_SIZE;
			}

		}
		else
		{
			step_length = DDA_STEP_SIZE;
		}

		//        
		point_x += direction_x * step_length;
		point_y += direction_y * step_length;
		point_z += direction_z * step_length;
		ray_length += step_length;

		//       Block     
		is_allocated = check_block_allocated(point_x, point_y, point_z, entry);
	}
	

	//             
	if (find_small_value)
	{
		bool valid_point = true;
		for (int i = 0; i < 8; i++)
		{
			valid_point = valid_point && get_sdf_interpolated(point_x, point_y, point_z, entry, voxel_block_array, sdf);

			//
			step_length = sdf * TRUNCATED_BAND;
			point_x += direction_x * step_length;
			point_y += direction_y * step_length;
			point_z += direction_z * step_length;
			ray_length += step_length;
		}
		

		if (valid_point)
		{
			// 
			int point_index = u + image_W * v;
			
			raycast_points[point_index].x = (float)(ray_length * camera_direction_x);
			raycast_points[point_index].y = (float)(ray_length * camera_direction_y);
			raycast_points[point_index].z = (float)(ray_length * camera_direction_z);

#if(0) /* Use  */
			//      
			float sdf_x, sdf_y, sdf_z, sdf_gradient;
			float point_x_inc, point_y_inc, point_z_inc;
			point_x_inc = point_x + VOXEL_SIZE * 1.0;
			point_y_inc = point_y + VOXEL_SIZE * 1.0;
			point_z_inc = point_z + VOXEL_SIZE * 1.0;

			//   SDF       
			valid_point &= get_sdf_interpolated(point_x, point_y, point_z, entry, voxel_block_array, sdf);
			valid_point &= get_sdf_interpolated(point_x_inc, point_y, point_z, entry, voxel_block_array, sdf_x);
			valid_point &= get_sdf_interpolated(point_x, point_y_inc, point_z, entry, voxel_block_array, sdf_y);
			valid_point &= get_sdf_interpolated(point_x, point_y, point_z_inc, entry, voxel_block_array, sdf_z);

			if (valid_point)
			{
				// Compute gradient
				sdf_x -= sdf;			sdf_y -= sdf;			sdf_z -= sdf;
				sdf_gradient = norm3df(sdf_x, sdf_y, sdf_z);
				sdf_x /= sdf_gradient;
				sdf_y /= sdf_gradient;
				sdf_z /= sdf_gradient;

				// 
				float norm_camera_x, norm_camera_y, norm_camera_z;
				norm_camera_x = camera_pose[0] * sdf_x + camera_pose[1] * sdf_y + camera_pose[2] * sdf_z;
				norm_camera_y = camera_pose[4] * sdf_x + camera_pose[5] * sdf_y + camera_pose[6] * sdf_z;
				norm_camera_z = camera_pose[8] * sdf_x + camera_pose[9] * sdf_y + camera_pose[10] * sdf_z;

				// Normalize normal vector
				raycast_normal[point_index].x = norm_camera_x;
				raycast_normal[point_index].y = norm_camera_y;
				raycast_normal[point_index].z = norm_camera_z;

				// Get weight
				//int voxel_index = find_voxel_index_round(point_x, point_y, point_z, entry);
				int voxel_index = get_voxel_index_neighbor(point_x, point_y, point_z, entry);
				raycast_weight[point_index] = voxel_block_array[voxel_index].weight;
				// Get plane label
				raycast_plane_label[point_index] = voxel_block_array[voxel_index].plane_index;
			}
			else
			{
				raycast_weight[point_index] = 0;
				raycast_plane_label[point_index] = 0;
			}
#else
			My_Type::Vector3f normal_vec(0.0f);

			// Interpolate normal vector
			valid_point &= interpolate_normal_by_sdf(point_x, point_y, point_z, entry, voxel_block_array, normal_vec);

			if (valid_point)
			{
				// 
				float norm_camera_x, norm_camera_y, norm_camera_z;
				norm_camera_x = camera_pose[0] * normal_vec.x + camera_pose[1] * normal_vec.y + camera_pose[2] * normal_vec.z;
				norm_camera_y = camera_pose[4] * normal_vec.x + camera_pose[5] * normal_vec.y + camera_pose[6] * normal_vec.z;
				norm_camera_z = camera_pose[8] * normal_vec.x + camera_pose[9] * normal_vec.y + camera_pose[10] * normal_vec.z;

				// Normalize normal vector
				raycast_normal[point_index].x = norm_camera_x;
				raycast_normal[point_index].y = norm_camera_y;
				raycast_normal[point_index].z = norm_camera_z;

				// Get weight
				//int voxel_index = find_voxel_index_round(point_x, point_y, point_z, entry);
				int voxel_index = get_voxel_index_neighbor(point_x, point_y, point_z, entry);
				raycast_weight[point_index] = voxel_block_array[voxel_index].weight;
				// Get plane label
				raycast_plane_label[point_index] = voxel_block_array[voxel_index].plane_index;
			}
			else
			{
				raycast_weight[point_index] = 0;
				raycast_plane_label[point_index] = 0;
			}

#endif
		}
	}
	

}
// Cpp call CUDA 
void raycast_byStep_CUDA(dim3 block_rect, dim3 thread_rect, 
						 const float * camera_pose, 
						 Sensor_params sensor_params, 
						 int raycast_patch_width,
						 const HashEntry * entry, 
						 const Voxel_f * voxel_block_array,
						 const My_Type::Vector2f * range_map, 
						 My_Type::Vector3f * raycast_points, 
						 My_Type::Vector3f * raycast_normal, 
						 int * raycast_plane_label, 
						 int * raycast_weight)
{
	raycast_byStep_KernelFunc << <block_rect, thread_rect >> >(camera_pose, sensor_params, raycast_patch_width,
															   entry, voxel_block_array, range_map,
															   raycast_points, raycast_normal, raycast_plane_label, raycast_weight);
}




// Direct fusion
__global__ void prj_fusion_sdf_KernelFunc(const My_Type::Vector3f * current_points, 
										  const float * camera_pose_inv,
										  Sensor_params sensor_params,
										  int depth_width, int depth_height,
										  const HashEntry * visible_list, 
										  Voxel_f * voxel_block_array)
{
	int block_pointer, voxel_offset;

	// block, voxel    （  ： ）
	float block_x, block_y, block_z, Vx, Vy, Vz;
	//        voxel    （  ： ）
	float camera_Vx, camera_Vy, camera_Vz;
	//         
	int prj_u, prj_v;
	// 
	float sdf_current, sdf_model, depth;
	int weight;

	
	//   Block  （  ： ）
	block_pointer = visible_list[blockIdx.x].ptr;
	block_x = ((float)visible_list[blockIdx.x].position[0]) * DDA_STEP_SIZE;
	block_y = ((float)visible_list[blockIdx.x].position[1]) * DDA_STEP_SIZE;
	block_z = ((float)visible_list[blockIdx.x].position[2]) * DDA_STEP_SIZE;
	

	//    Voxel   （ ）
	Vx = block_x + (float)threadIdx.x * VOXEL_SIZE;
	Vy = block_y + (float)threadIdx.y * VOXEL_SIZE;
	Vz = block_z + (float)threadIdx.z * VOXEL_SIZE;

	//        voxel   
	camera_Vx = camera_pose_inv[0] * Vx + camera_pose_inv[4] * Vy + camera_pose_inv[8] * Vz + camera_pose_inv[12];
	camera_Vy = camera_pose_inv[1] * Vx + camera_pose_inv[5] * Vy + camera_pose_inv[9] * Vz + camera_pose_inv[13];
	camera_Vz = camera_pose_inv[2] * Vx + camera_pose_inv[6] * Vy + camera_pose_inv[10] * Vz + camera_pose_inv[14];
	//    
	camera_Vx /= camera_Vz;		camera_Vy /= camera_Vz;

	//         
	prj_u = (int)(sensor_params.sensor_fx * camera_Vx + sensor_params.sensor_cx);
	prj_v = (int)(sensor_params.sensor_fy * camera_Vy + sensor_params.sensor_cy);
	//           
	if (prj_u < 0 || prj_u >= depth_width)	return;
	if (prj_v < 0 || prj_v >= depth_height)	return;
	

	//      
	depth = (float)current_points[prj_u + prj_v * depth_width].z;
	//         
	//if (depth < MIN_VALID_DEPTH_M || depth > MAX_VALID_DEPTH_M)		return;
	if (depth < FLT_EPSILON)		return;


	//      SDF
	sdf_current = (depth - camera_Vz) / TRUNCATED_BAND;
	//       
	if ((sdf_current > 1.0f) || (sdf_current < -1.0f))	return;

	//     
	voxel_offset = threadIdx.x + threadIdx.y * VOXEL_BLOCK_WDITH + threadIdx.z * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH;
	weight = voxel_block_array[block_pointer + voxel_offset].weight;
	sdf_model = voxel_block_array[block_pointer + voxel_offset].sdf;

	

	//    TSDF 
	voxel_block_array[block_pointer + voxel_offset].sdf = (sdf_model * (float)weight + sdf_current) / (weight + 1.0f);
	if (weight < MAX_SDF_WEIGHT)
	{
		voxel_block_array[block_pointer + voxel_offset].weight = weight + 1;
	}
}
// Cpp call CUDA
void prj_fusion_sdf_CUDA(dim3 block_rect, dim3 thread_rect, 
						 const My_Type::Vector3f * current_points,
						 const float * camera_pose_inv,
						 Sensor_params sensor_params,
						 int depth_width, int depth_height,
						 const HashEntry * visible_list,
						 Voxel_f * voxel_block_array)
{

	prj_fusion_sdf_KernelFunc << <block_rect, thread_rect >> >(current_points, camera_pose_inv, sensor_params, depth_width, depth_height, visible_list, 
															   voxel_block_array);
}

//
__global__ void prj_normal_fusion_sdf_KernelFunc(const My_Type::Vector3f * current_points, 
												 const My_Type::Vector3f * current_normals, 
												 const float * camera_pose_inv,
												 Sensor_params sensor_params,
												 int depth_width, int depth_height,
												 const HashEntry * visible_list,
												 Voxel_f * voxel_block_array)
{
	int block_pointer, voxel_offset;

	//
	My_Type::Vector3f block_in_world, voxel_in_world, voxel_in_camera;


	// 
	int prj_u, prj_v;
	// 
	int weight;


	//   Block  （  ： ）
	block_pointer = visible_list[blockIdx.x].ptr;
	block_in_world.x = ((float)visible_list[blockIdx.x].position[0]) * DDA_STEP_SIZE;
	block_in_world.y = ((float)visible_list[blockIdx.x].position[1]) * DDA_STEP_SIZE;
	block_in_world.z = ((float)visible_list[blockIdx.x].position[2]) * DDA_STEP_SIZE;


	//    Voxel   （ ）
	voxel_in_world.x = block_in_world.x + (float)threadIdx.x * VOXEL_SIZE;
	voxel_in_world.y = block_in_world.y + (float)threadIdx.y * VOXEL_SIZE;
	voxel_in_world.z = block_in_world.z + (float)threadIdx.z * VOXEL_SIZE;


	//        voxel   
	voxel_in_camera.x = camera_pose_inv[0] * voxel_in_world.x + camera_pose_inv[4] * voxel_in_world.y + camera_pose_inv[8] * voxel_in_world.z + camera_pose_inv[12];
	voxel_in_camera.y = camera_pose_inv[1] * voxel_in_world.x + camera_pose_inv[5] * voxel_in_world.y + camera_pose_inv[9] * voxel_in_world.z + camera_pose_inv[13];
	voxel_in_camera.z = camera_pose_inv[2] * voxel_in_world.x + camera_pose_inv[6] * voxel_in_world.y + camera_pose_inv[10] * voxel_in_world.z + camera_pose_inv[14];

	//         
	prj_u = (int)(sensor_params.sensor_fx * (voxel_in_camera.x / voxel_in_camera.z) + sensor_params.sensor_cx);
	prj_v = (int)(sensor_params.sensor_fy * (voxel_in_camera.y / voxel_in_camera.z) + sensor_params.sensor_cy);
	//           
	if (prj_u < 0 || prj_u >= depth_width)	return;
	if (prj_v < 0 || prj_v >= depth_height)	return;

	//      
	My_Type::Vector3f current_point = current_points[prj_u + prj_v * depth_width];
	if (current_point.z < FLT_EPSILON)		return;

	// Compute cos(\theta)
	My_Type::Vector3f current_normal, ray_vector;
	if (voxel_in_camera.norm() < FLT_EPSILON)	return;
	ray_vector = voxel_in_camera;
	ray_vector /= voxel_in_camera.norm();
	current_normal = current_normals[prj_u + prj_v * depth_width];
	float inner_product = - ray_vector.x * current_normal.x - ray_vector.y * current_normal.y - ray_vector.z * current_normal.z;
	if (inner_product < 0.5f)	inner_product = 0.5f;


	// Diff distance
	float diff_distance = (current_point - voxel_in_camera).norm();
	if (current_point.z - voxel_in_camera.z  < 0)		diff_distance = -diff_distance;
	float sdf_current = diff_distance * inner_product / TRUNCATED_BAND;
	//       
	if ((sdf_current > 1.0f) || (sdf_current < -1.0f))	return;

	//     
	voxel_offset = threadIdx.x + threadIdx.y * VOXEL_BLOCK_WDITH + threadIdx.z * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH;
	weight = voxel_block_array[block_pointer + voxel_offset].weight;
	float sdf_model = voxel_block_array[block_pointer + voxel_offset].sdf;

	//    TSDF 
	voxel_block_array[block_pointer + voxel_offset].sdf = (sdf_model * (float)weight + sdf_current) / (weight + 1.0f);
	if (weight < MAX_SDF_WEIGHT)
	{
		voxel_block_array[block_pointer + voxel_offset].weight = weight + 1;
	}
}
// Cpp call CUDA
void prj_normal_fusion_sdf_CUDA(dim3 block_rect, dim3 thread_rect, 
								const My_Type::Vector3f * current_points,
								const My_Type::Vector3f * current_normal,
								const float * camera_pose_inv,
								Sensor_params sensor_params,
								int depth_width, int depth_height,
								const HashEntry * visible_list,
								Voxel_f * voxel_block_array)
{
	prj_normal_fusion_sdf_KernelFunc << <block_rect, thread_rect >> >(current_points, current_normal, camera_pose_inv, sensor_params, depth_width, depth_height, visible_list,
																	  voxel_block_array);
}



// 
__global__ void fusion_map_sdf_KernelFunc(HashEntry * relative_list, int * relative_counter, Voxel_f * this_Voxel_map,  float * pose_inv, \
	HashEntry * fragment_entry, Voxel_f * fragment_Voxel_map, int * plane_global_index)
{
	//
	int block_index = blockIdx.x;
	//    Entry
	HashEntry map_entry = relative_list[block_index];
	// 
	bool is_valid_voxel = true;

	//
	if (block_index >= relative_counter[0])	is_valid_voxel = false;
	 
	//
	My_Type::Vector3f block_position;
	block_position.x = (float)map_entry.position[0];
	block_position.y = (float)map_entry.position[1];
	block_position.z = (float)map_entry.position[2];
	block_position *= (VOXEL_SIZE * VOXEL_BLOCK_WDITH);
	//if (map_entry.position[0] == 0 && map_entry.position[1] == 0 && map_entry.position[2] == 0)	is_valid_voxel = false;

	//   Voxel  
	My_Type::Vector3f voxel_position;
	voxel_position = block_position;
	voxel_position.x += VOXEL_SIZE * (float)threadIdx.x;
	voxel_position.y += VOXEL_SIZE * (float)threadIdx.y;
	voxel_position.z += VOXEL_SIZE * (float)threadIdx.z;

	//   Voxel   Fragment    
	My_Type::Vector3f voxel_position_f;
	voxel_position_f.x = pose_inv[0] * voxel_position.x + pose_inv[4] * voxel_position.y + pose_inv[8]	* voxel_position.z + pose_inv[12];
	voxel_position_f.y = pose_inv[1] * voxel_position.x + pose_inv[5] * voxel_position.y + pose_inv[9]	* voxel_position.z + pose_inv[13];
	voxel_position_f.z = pose_inv[2] * voxel_position.x + pose_inv[6] * voxel_position.y + pose_inv[10] * voxel_position.z + pose_inv[14];
	//voxel_position_f = voxel_position;

	//        Voxel 
	//int voxel_index_f = find_voxel_index_round(voxel_position_f.x, voxel_position_f.y, voxel_position_f.z, fragment_entry);
	int voxel_index_f = get_voxel_index_neighbor(voxel_position_f.x, voxel_position_f.y, voxel_position_f.z, fragment_entry);
	if (voxel_index_f < 0)	is_valid_voxel = false;
	
	//
	int voxel_offset, voxel_index;
	Voxel_f model_voxel, fragment_voxel;
	if (is_valid_voxel)
	{
		//
		fragment_voxel = fragment_Voxel_map[voxel_index_f];

		if (fragment_voxel.weight < MIN_RAYCAST_WEIGHT)	is_valid_voxel = false;
	}

	//      SDF 
	float interpolate_sdf;
	if (is_valid_voxel)
	{
		//     Pose      Vx、Vy、Vz      
		voxel_position_f += FLT_EPSILON * 10;
		//voxel_position_f += FLT_EPSILON * 1;
		//         SDF ，   VoxelBlock    
		is_valid_voxel = get_sdf_interpolated(voxel_position_f.x, voxel_position_f.y, voxel_position_f.z, \
			fragment_entry, fragment_Voxel_map, interpolate_sdf);
	}

	
	//     
	int block_ptr = map_entry.ptr;
	if (is_valid_voxel)
	{
		// Voxel   
		voxel_offset = threadIdx.x + threadIdx.y * VOXEL_BLOCK_WDITH + threadIdx.z *  VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH;
		voxel_index = block_ptr + voxel_offset;
		model_voxel = this_Voxel_map[voxel_index];
		
		//     
		model_voxel.sdf = (model_voxel.sdf * (float)model_voxel.weight + interpolate_sdf * (float)fragment_voxel.weight) / \
			(float)(model_voxel.weight + fragment_voxel.weight);
		//model_voxel.sdf = interpolate_sdf;
		if (model_voxel.plane_index == 0)
		{
			model_voxel.plane_index = plane_global_index[fragment_voxel.plane_index];
		}		
		model_voxel.weight = min(MAX_SDF_WEIGHT, model_voxel.weight + fragment_voxel.weight);
		this_Voxel_map[voxel_index] = model_voxel;

		//// Debug:
		//int debug_offset = voxel_index_f % VOXEL_BLOCK_SIZE;
		//if (debug_offset != voxel_offset && true)
		//{
		//	int offset_x, offset_y, offset_z, offset_linear;
		//	int block_pos[3];
		//	// Block    
		//	block_pos[0] = floorf(voxel_position_f.x / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
		//	block_pos[1] = floorf(voxel_position_f.y / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
		//	block_pos[2] = floorf(voxel_position_f.z / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
		//	// Voxel   Block     
		//	offset_x = roundf((voxel_position_f.x - ((float)block_pos[0]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
		//	offset_y = roundf((voxel_position_f.y - ((float)block_pos[1]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
		//	offset_z = roundf((voxel_position_f.z - ((float)block_pos[2]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
		//	printf("%d, %d, %d, %d, %d, %d\r\n", threadIdx.x, offset_x, threadIdx.y, offset_y, threadIdx.z, offset_z);
		//}
	}

}
//! CPP call CUDA
void fusion_map_sdf_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * relative_list, int * relative_counter, Voxel_f * this_Voxel_map, \
	float * pose_inv, HashEntry * fragment_entry, Voxel_f * fragment_Voxel_map, int * plane_global_index)
{

	fusion_map_sdf_KernelFunc << <block_rect, thread_rect >> >(relative_list, relative_counter, this_Voxel_map, pose_inv, \
		fragment_entry, fragment_Voxel_map, plane_global_index);
}




//       
//           voxel
__global__ void prj_fusion_plane_label_KernelFunc(const My_Type::Vector3f * current_points, 
												  const float * camera_pose_inv, 
												  Sensor_params sensor_params,
												  int depth_width, int depth_height,
												  const HashEntry * visible_list, 
												  Voxel_f * voxel_block_array, 
												  int * plane_img)
{
	int block_pointer, voxel_offset;

	// block, voxel    （  ： ）
	float block_x, block_y, block_z, Vx, Vy, Vz;
	//        voxel    （  ： ）
	float camera_Vx, camera_Vy, camera_Vz;
	//         
	int prj_u, prj_v;
	// 
	float sdf_current, depth;


	//   Block  （  ： ）
	block_pointer = visible_list[blockIdx.x].ptr;
	block_x = ((float)visible_list[blockIdx.x].position[0]) * DDA_STEP_SIZE;
	block_y = ((float)visible_list[blockIdx.x].position[1]) * DDA_STEP_SIZE;
	block_z = ((float)visible_list[blockIdx.x].position[2]) * DDA_STEP_SIZE;


	//    Voxel   （ ）
	Vx = block_x + (float)threadIdx.x * VOXEL_SIZE;
	Vy = block_y + (float)threadIdx.y * VOXEL_SIZE;
	Vz = block_z + (float)threadIdx.z * VOXEL_SIZE;

	//        voxel   
	camera_Vx = camera_pose_inv[0] * Vx + camera_pose_inv[4] * Vy + camera_pose_inv[8] * Vz + camera_pose_inv[12];
	camera_Vy = camera_pose_inv[1] * Vx + camera_pose_inv[5] * Vy + camera_pose_inv[9] * Vz + camera_pose_inv[13];
	camera_Vz = camera_pose_inv[2] * Vx + camera_pose_inv[6] * Vy + camera_pose_inv[10] * Vz + camera_pose_inv[14];
	//    
	camera_Vx /= camera_Vz;		camera_Vy /= camera_Vz;

	//         
	prj_u = (int)(sensor_params.sensor_fx * camera_Vx + sensor_params.sensor_cx);
	prj_v = (int)(sensor_params.sensor_fy * camera_Vy + sensor_params.sensor_cy);
	//           
	if (prj_u < 0 || prj_u >= depth_width)	return;
	if (prj_v < 0 || prj_v >= depth_height)	return;


	//      
	depth = (float)current_points[prj_u + prj_v * depth_width].z;
	//         
	//if (depth < MIN_VALID_DEPTH_M || depth > MAX_VALID_DEPTH_M)		return;
	if (depth < FLT_EPSILON)		return;

	//      SDF
	sdf_current = (depth - camera_Vz) / TRUNCATED_BAND;
	//       
	if ((sdf_current > 1.0f) || (sdf_current < -1.0f))	return;

	//     
	voxel_offset = threadIdx.x + threadIdx.y * VOXEL_BLOCK_WDITH + threadIdx.z * VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH;


	//       
	//     
	int plane_index = plane_img[prj_u + prj_v * depth_width];
	if (plane_index > 0)
	{
		voxel_block_array[block_pointer + voxel_offset].plane_index = (unsigned short)plane_index;
	}
}
// Cpp call CUDA
void prj_fusion_plane_label_CUDA(dim3 block_rect, dim3 thread_rect, 
								 const My_Type::Vector3f * current_points,
								 const float * camera_pose_inv,
								 Sensor_params sensor_params,
								 int depth_width, int depth_height,
								 const HashEntry * visible_list,
								 Voxel_f * voxel_block_array,
								 int * plane_img)
{
	prj_fusion_plane_label_KernelFunc << <block_rect, thread_rect >> >(current_points, camera_pose_inv, sensor_params, depth_width, depth_height, visible_list,
																	   voxel_block_array, plane_img);
}



//     Block              
__global__ void reduce_range_KernelFunc(const My_Type::Vector3i * __restrict__ block_position, 
										int block_num, 
										int * min_depth, int * max_depth, 
										const float * pose_inv)
{
	// 
	bool is_valid_block = true;
	//     
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= block_num)		is_valid_block = false;


	float current_depth;
	if (is_valid_block)
	{
		// Block   
		My_Type::Vector3f position;
		position.x = ((float)block_position[index].x) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
		position.y = ((float)block_position[index].y) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;
		position.z = ((float)block_position[index].z) * VOXEL_BLOCK_WDITH * VOXEL_SIZE;

		// Block   
		current_depth = pose_inv[2] * position.x + pose_inv[6] * position.y + pose_inv[10] * position.z + pose_inv[14];
		//      
		if (current_depth < FLT_EPSILON)		is_valid_block = false;
	}


	// Reduce minimum and maximum
	int tid = threadIdx.x;
	__shared__ float min_cache[256], max_cache[256];
	if (is_valid_block)
	{
		min_cache[tid] = current_depth;
		max_cache[tid] = current_depth;
	}
	else
	{
		min_cache[tid] = FLT_MAX;
		max_cache[tid] = 0;
	}
	//     
	block_256_reduce_min(min_cache, tid);
	block_256_reduce_max(max_cache, tid);


	// 
	if (tid == 0)
	{
		min_cache[0] -= DDA_STEP_SIZE;
		min_cache[0] = max(min_cache[0], 0.0f);
		atomicMin(min_depth, (int)(1000.0f * min_cache[0]));
		max_cache[0] += DDA_STEP_SIZE;
		atomicMax(max_depth, (int)(1000.0f * max_cache[0]));
	}
}
// Cpp call CUDA
void reduce_range_CUDA(dim3 block_rect, dim3 thread_rect, 
					   const My_Type::Vector3i * __restrict__ block_position, 
					   int block_num,
					   int * min_depth, int * max_depth, 
					   const float * pose_inv)
{

	reduce_range_KernelFunc << <block_rect, thread_rect >> >(block_position, block_num, min_depth, max_depth, pose_inv);
}


// Merge depth with occlusion judgment
__global__ void merge_frame_with_occlusion_KernelFunc(My_Type::Vector3f * global_points, My_Type::Vector3f * fragment_points, \
	My_Type::Vector3f * merge_points, My_Type::Vector3f * global_normal, My_Type::Vector3f * fragment_normal, My_Type::Vector3f * merge_normal, \
	int * global_weight, int * fragment_weight, int * merge_weight, int * global_plane_img, int * fragment_plane_img, int * merge_plane_img)
{
	//
	int u, v, index, image_width;
	u = threadIdx.x + blockDim.x * blockIdx.x;
	v = threadIdx.y + blockDim.y * blockIdx.y;
	//
	image_width = blockDim.x * gridDim.x;
	index = u + v * image_width;

	//
	bool is_global_point_valid = true;
	bool is_fragment_point_valid = true;
	//
	bool is_global_point = false;
	bool is_fragment_point = false;


	// Read both farmes' depth
	float global_depth = global_points[index].z;
	float fragment_depth = fragment_points[index].z;
	//if (global_depth < FLT_EPSILON || global_depth > MAX_VALID_DEPTH_M)			is_global_point_valid = false;
	//if (fragment_depth < FLT_EPSILON || fragment_depth > MAX_VALID_DEPTH_M)		is_fragment_point_valid = false;
	if (global_depth < FLT_EPSILON)		return;
	if (fragment_depth < FLT_EPSILON)		return;



	// both valid, compute occlusion
	if (is_global_point_valid && is_fragment_point_valid)
	{
		//  compute occlusion
		if (global_depth < fragment_depth)
		{
			is_global_point = true;
		}
		else
		{
			is_fragment_point = true;
		}
	}
	// either global or fragment detected
	if (is_global_point_valid && (!is_fragment_point_valid))	is_global_point = true;
	if ((!is_global_point_valid) && is_fragment_point_valid)	is_fragment_point = true;


	// save result
	if (is_global_point)
	{
		merge_points[index] = global_points[index];
		merge_normal[index] = global_normal[index];
		merge_weight[index] = global_weight[index];
		merge_plane_img[index] = global_plane_img[index];
	}
	else if (is_fragment_point)
	{
		merge_points[index] = fragment_points[index];
		merge_normal[index] = fragment_normal[index];
		merge_weight[index] = fragment_weight[index];
		merge_plane_img[index] = fragment_plane_img[index];
	}
	else
	{
		My_Type::Vector3f zero_vec3(0, 0, 0);
		// set to empty
		merge_points[index] = zero_vec3;
		merge_normal[index] = zero_vec3;
		merge_weight[index] = 0;
		merge_plane_img[index] = 0;
	}

}
// Cpp call CUDA
void merge_frame_with_occlusion_CUDA(dim3 block_rect, dim3 thread_rect, My_Type::Vector3f * global_points, My_Type::Vector3f * fragment_points, \
	My_Type::Vector3f * merge_points, My_Type::Vector3f * global_normal, My_Type::Vector3f * fragment_normal, My_Type::Vector3f * merge_normal, \
	int * global_weight, int * fragment_weight, int * merge_weight, int * global_plane_img, int * fragment_plane_img, int * merge_plane_img)
{

	merge_frame_with_occlusion_KernelFunc << <block_rect, thread_rect >> >(global_points, fragment_points, merge_points, global_normal, fragment_normal, \
		merge_normal, global_weight, fragment_weight, merge_weight, global_plane_img, fragment_plane_img, merge_plane_img);
}



// for debug copy voxel map
__global__ void copy_voexl_map_KernelFunc(HashEntry * relative_list, int * relative_counter, Voxel_f * dst_Voxel_map, \
	HashEntry * fragment_entry, Voxel_f * fragment_Voxel_map)
{
	//
	int block_index = blockIdx.x;
	//    Entry
	HashEntry map_entry = relative_list[block_index];
	//
	bool is_valid_voxel = true;

	//
	My_Type::Vector3i block_pos;
	block_pos.x = map_entry.position[0];
	block_pos.y = map_entry.position[1];
	block_pos.z = map_entry.position[2];

	//   Block   Entry
	int entry_index;
	if (is_valid_voxel)
	{
		bool is_find = false;
		int hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);

		//  ordered    
		if ((fragment_entry[hash_value].position[0] == block_pos[0]) && \
			(fragment_entry[hash_value].position[1] == block_pos[1]) && \
			(fragment_entry[hash_value].position[2] == block_pos[2]))
		{
			is_find = true;
			entry_index = hash_value;
		}
		// ordered     
		if (!is_find)
		{
			int excess_offset = fragment_entry[hash_value].offset;

			//  excess    
			while (excess_offset >= 0)
			{
				//       
				if ((fragment_entry[excess_offset].position[0] == block_pos[0]) && \
					(fragment_entry[excess_offset].position[1] == block_pos[1]) && \
					(fragment_entry[excess_offset].position[2] == block_pos[2]))
				{
					is_find = true;
					entry_index = excess_offset;
					break;
				}

				//      
				excess_offset = fragment_entry[excess_offset].offset;
			}
		}

		//
		if (!is_find)	is_valid_voxel = false;
	}


	//   Voxel 
	if (is_valid_voxel)
	{
		int block_ptr = fragment_entry[entry_index].ptr;
		int voxel_offset = threadIdx.x + threadIdx.y * VOXEL_BLOCK_WDITH + threadIdx.z *  VOXEL_BLOCK_WDITH * VOXEL_BLOCK_WDITH;
		//
		int voxel_index = block_ptr + voxel_offset;

		//     Voxel 
		Voxel_f src_voxel = fragment_Voxel_map[voxel_index];
		//   
		int dst_block_ptr = map_entry.ptr;
		int dst_voxel_index = dst_block_ptr + voxel_offset;
		dst_Voxel_map[dst_voxel_index] = src_voxel;
	}
}
// Cpp call CUDA
void copy_voexl_map_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * relative_list, int * relative_counter, Voxel_f * dst_Voxel_map, \
	HashEntry * fragment_entry, Voxel_f * fragment_Voxel_map)
{

	copy_voexl_map_KernelFunc << <block_rect, thread_rect >> >(relative_list, relative_counter, dst_Voxel_map, fragment_entry, fragment_Voxel_map);
}


// Reduce the weight center of current voxel block
__global__ void reduce_current_voxel_block_position_KernelFunc(HashEntry * visible_list, My_Type::Vector3f * current_weight_center, int visible_counter)
{
	//
	bool is_valid_entry = true;

	// Index of visible list
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	// Check if index out of range
	if (index >= visible_counter)	is_valid_entry = false;


	// Reduce sum
	__shared__ float cache_f[256];
	int tid = threadIdx.x;
	for (int i = 0; i < 3; i++)
	{
		if (is_valid_entry)
		{
			cache_f[tid] = (float)visible_list[index].position[i] * VOXEL_SIZE * VOXEL_BLOCK_WDITH;
		}
		else
		{
			cache_f[tid] = 0;
		}
		block_256_reduce(cache_f, tid);

		//
		if (tid == 0)
		{
			switch (i)
			{
				case 0:
				{
					atomicAdd((float *)&(current_weight_center[0].x), cache_f[0]);
					break;
				}
				case 1:
				{
					atomicAdd((float *)&(current_weight_center[0].y), cache_f[0]);
					break;
				}
				case 2:
				{
					atomicAdd((float *)&(current_weight_center[0].z), cache_f[0]);
					break;
				}
			default:
				break;
			}
		}
	}

	
}
// Cpp call CUDA
void reduce_current_voxel_block_position_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * visible_list, \
	My_Type::Vector3f * current_weight_center, int visible_counter)
{

	reduce_current_voxel_block_position_KernelFunc << <block_rect, thread_rect >> >(visible_list, current_weight_center, visible_counter);
}



// Reduce the (weight center/min point/max point) of Map's voxel block
__global__ void reduce_map_voxel_block_position_KernelFunc(HashEntry * entry, My_Type::Vector3f * map_weight_center)
{
	//
	bool is_valid_entry = true;

	// Compute Hash value
	int hash_value = threadIdx.x + blockDim.x * blockIdx.x;
	// Check entry is allocated
	if (entry[hash_value].ptr < 0)	is_valid_entry = false;


	// Reduce sum
	__shared__ float cache_f[256];
	int tid = threadIdx.x;
	for (int i = 0; i < 3; i++)
	{
		if (is_valid_entry)
		{
			cache_f[tid] = (float)entry[hash_value].position[i] * VOXEL_SIZE * VOXEL_BLOCK_WDITH;
		}
		else
		{
			cache_f[tid] = 0;
		}
		block_256_reduce(cache_f, tid);

		//
		if (tid == 0)
		{
			switch (i)
			{
				case 0:
				{
					atomicAdd(&(map_weight_center[0].x), cache_f[0]);
					break;
				}
				case 1:
				{
					atomicAdd(&(map_weight_center[0].y), cache_f[0]);
					break;
				}
				case 2:
				{
					atomicAdd(&(map_weight_center[0].z), cache_f[0]);
					break;
				}
				default:
					break;
			}
		}
	}

}
// Cpp call CUDA
void reduce_map_voxel_block_position_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * entry, My_Type::Vector3f * map_weight_center)
{

	reduce_map_voxel_block_position_KernelFunc << <block_rect, thread_rect >> >(entry, map_weight_center);
}


// Reduce fragment map's bounding box
__global__ void reduce_map_bounding_box_KernelFunc(HashEntry * entry, My_Type::Vector3i * map_min_offset, My_Type::Vector3i * map_max_offset)
{
	//
	bool is_valid_entry = true;

	// Compute Hash value
	int hash_value = threadIdx.x + blockDim.x * blockIdx.x;
	// Check entry is allocated
	if (entry[hash_value].ptr < 0)	is_valid_entry = false;

	// Read voxel block position
	__shared__ int cache_i1[256], cache_i2[256];
	int tid = threadIdx.x;
	for (int i = 0; i < 3; i++)
	{
		if (is_valid_entry)
		{
			cache_i1[tid] = entry[hash_value].position[i];
			cache_i2[tid] = entry[hash_value].position[i];
		}
		else
		{
			cache_i1[tid] = INT_MAX;
			cache_i2[tid] = INT_MIN;
		}
		block_256_reduce_min(cache_i1, tid);
		block_256_reduce_max(cache_i2, tid);

		//
		if (tid == 0)
		{
			switch (i)
			{
			case 0:
			{
				atomicMin(&(map_min_offset[0].x), cache_i1[0]);
				atomicMax(&(map_max_offset[0].x), cache_i2[0]);
				break;
			}
			case 1:
			{
				atomicMin(&(map_min_offset[0].y), cache_i1[0]);
				atomicMax(&(map_max_offset[0].y), cache_i2[0]);
				break;
			}
			case 2:
			{
				atomicMin(&(map_min_offset[0].z), cache_i1[0]);
				atomicMax(&(map_max_offset[0].z), cache_i2[0]);
				break;
			}
			default:
				break;
			}
		}
	}


}
// Cpp call CUDA
void reduce_map_bounding_box_CUDA(dim3 block_rect, dim3 thread_rect, HashEntry * entry, My_Type::Vector3i * map_min_offset, My_Type::Vector3i * map_max_offset)
{

	reduce_map_bounding_box_KernelFunc << <block_rect, thread_rect >> >(entry, map_min_offset, map_max_offset);
}



//        Block    
__device__ inline bool check_block_allocated(float & Vx, float & Vy, float & Vz, const HashEntry * entry)
{
	int block_pos[3];

	//   Hash 
	float voxel_size = VOXEL_SIZE;
	block_pos[0] = floorf(round_by_stride(Vx, voxel_size) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[1] = floorf(round_by_stride(Vy, voxel_size) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[2] = floorf(round_by_stride(Vz, voxel_size) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));

	bool is_find = false;
	int hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);

	//  ordered    
	if ((entry[hash_value].position[0] == block_pos[0]) && \
		(entry[hash_value].position[1] == block_pos[1]) && \
		(entry[hash_value].position[2] == block_pos[2]))
	{
		is_find = true;
	}
	// ordered     
	if (!is_find)
	{
		int excess_offset = entry[hash_value].offset;

		//  excess    
		while (excess_offset >= 0)
		{
			//       
			if ((entry[excess_offset].position[0] == block_pos[0]) && \
				(entry[excess_offset].position[1] == block_pos[1]) && \
				(entry[excess_offset].position[2] == block_pos[2]))
			{
				is_find = true;
				break;
			}

			//      
			excess_offset = entry[excess_offset].offset;
		}
	}

	return is_find;
}
//   
__device__ inline bool check_block_allocated(int & block_pos_x, int & block_pos_y, int & block_pos_z, const HashEntry * entry)
{
	int block_pos[3];

	//   Hash 
	block_pos[0] = block_pos_x;
	block_pos[1] = block_pos_y;
	block_pos[2] = block_pos_z;


	bool is_find = false;
	int hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);

	//  ordered    
	if ((entry[hash_value].position[0] == block_pos[0]) && \
		(entry[hash_value].position[1] == block_pos[1]) && \
		(entry[hash_value].position[2] == block_pos[2]))
	{
		is_find = true;
	}
	// ordered     
	if (!is_find)
	{
		int excess_offset = entry[hash_value].offset;

		//  excess    
		while (excess_offset >= 0)
		{
			//       
			if ((entry[excess_offset].position[0] == block_pos[0]) && \
				(entry[excess_offset].position[1] == block_pos[1]) && \
				(entry[excess_offset].position[2] == block_pos[2]))
			{
				is_find = true;
				break;
			}

			//      
			excess_offset = entry[excess_offset].offset;
		}
	}

	return is_find;
}


// Voxel
__device__ inline int find_voxel_index_round(float & Vx, float & Vy, float & Vz, const HashEntry * entry)
{
	int block_pos[3], hash_value;
	int offset_x, offset_y, offset_z, offset_linear;

	//
	block_pos[0] = floorf((Vx + HALF_VOXEL_SIZE) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[1] = floorf((Vy + HALF_VOXEL_SIZE) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[2] = floorf((Vz + HALF_VOXEL_SIZE) / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	
	//
	offset_x = roundf((Vx - ((float)block_pos[0]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
	offset_y = roundf((Vy - ((float)block_pos[1]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
	offset_z = roundf((Vz - ((float)block_pos[2]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE);
	

	//     
	offset_linear = offset_x + (offset_y + offset_z * VOXEL_BLOCK_WDITH) * VOXEL_BLOCK_WDITH;

	//     Block     
	bool is_find = false;
	hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);
	//  ordered    
	if ((entry[hash_value].position[0] == block_pos[0]) && \
		(entry[hash_value].position[1] == block_pos[1]) && \
		(entry[hash_value].position[2] == block_pos[2]))
	{
		is_find = true;
		offset_linear += entry[hash_value].ptr;
	}
	// ordered     
	if (!is_find)
	{
		int excess_offset = entry[hash_value].offset;

		//  excess    
		while (excess_offset >= 0)
		{
			//       
			if ((entry[excess_offset].position[0] == block_pos[0]) && \
				(entry[excess_offset].position[1] == block_pos[1]) && \
				(entry[excess_offset].position[2] == block_pos[2]))
			{
				is_find = true;
				offset_linear += entry[excess_offset].ptr;
				break;
			}

			//      
			excess_offset = entry[excess_offset].offset;
		}

	}

	// 
	if (!is_find)
	{
		offset_linear = -1;
	}


	return offset_linear;
}

//
__device__ inline int get_voxel_index_neighbor(float & Vx, float & Vy, float & Vz, const HashEntry * entry)
{
	int block_pos[3], hash_value;
	int offset_x, offset_y, offset_z, offset_linear;


	// Block    
	block_pos[0] = floorf(Vx / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[1] = floorf(Vy / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));
	block_pos[2] = floorf(Vz / (VOXEL_SIZE * VOXEL_BLOCK_WDITH));

	// Voxel   Block     
	offset_x = (Vx - ((float)block_pos[0]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE;
	offset_y = (Vy - ((float)block_pos[1]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE;
	offset_z = (Vz - ((float)block_pos[2]) * (VOXEL_SIZE * VOXEL_BLOCK_WDITH)) / VOXEL_SIZE;

	//     
	offset_linear = offset_x + (offset_y + offset_z * VOXEL_BLOCK_WDITH) * VOXEL_BLOCK_WDITH;

	//     Block     
	bool is_find = false;
	hash_value = hash_func(block_pos[0], block_pos[1], block_pos[2]);
	//  ordered    
	if ((entry[hash_value].position[0] == block_pos[0]) && \
		(entry[hash_value].position[1] == block_pos[1]) && \
		(entry[hash_value].position[2] == block_pos[2]))
	{
		is_find = true;
		offset_linear += entry[hash_value].ptr;
	}
	// ordered     
	if (!is_find)
	{
		int excess_offset = entry[hash_value].offset;

		//  excess    
		while (excess_offset >= 0)
		{
			//       
			if ((entry[excess_offset].position[0] == block_pos[0]) && \
				(entry[excess_offset].position[1] == block_pos[1]) && \
				(entry[excess_offset].position[2] == block_pos[2]))
			{
				is_find = true;
				offset_linear += entry[excess_offset].ptr;
				break;
			}

			//      
			excess_offset = entry[excess_offset].offset;
		}

	}

	// 
	if (!is_find)
	{
		offset_linear = -1;
	}


	return offset_linear;
}


//         
__device__ inline bool get_sdf_interpolated(float & Vx, float & Vy, float & Vz,
											const HashEntry * entry, const Voxel_f * voxel_block_array,
											float & sdf_interpolated)
{
	int voxel_index;
	float sdf_0, sdf_1, interpolate_x0, interpolate_x1, interpolate_y0, interpolate_y1;

	//     
	float step = VOXEL_SIZE;
	float coeff_x, coeff_y, coeff_z;
	coeff_x = (Vx + HALF_VOXEL_SIZE - floor_by_stride(Vx, step)) / step;
	coeff_y = (Vy + HALF_VOXEL_SIZE - floor_by_stride(Vy, step)) / step;
	coeff_z = (Vz + HALF_VOXEL_SIZE - floor_by_stride(Vz, step)) / step;


	float Px, Py, Pz, P1_x, P1_y, P1_z;
	P1_x = Vx - HALF_VOXEL_SIZE;
	P1_y = Vy - HALF_VOXEL_SIZE;
	P1_z = Vz - HALF_VOXEL_SIZE;
	// z y x
	// 0 0 0
	Pz = P1_z;	Py = P1_y;	Px = P1_x;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_0 = voxel_block_array[voxel_index].sdf;
	// z y x
	// 0 0 1
	Pz = P1_z;	Py = P1_y;	Px = P1_x + VOXEL_SIZE;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_1 = voxel_block_array[voxel_index].sdf;
	// - 0 0 x
	interpolate_x0 = sdf_0 * (1 - coeff_x) + sdf_1 * coeff_x;

	// z y x
	// 0 1 0
	Pz = P1_z;	Py = P1_y + VOXEL_SIZE;	Px = P1_x;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_0 = voxel_block_array[voxel_index].sdf;
	// z y x
	// 0 1 1
	Pz = P1_z;	Py = P1_y + VOXEL_SIZE;	Px = P1_x + VOXEL_SIZE;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_1 = voxel_block_array[voxel_index].sdf;
	// - 0 1 x
	interpolate_x1 = sdf_0 * (1 - coeff_x) + sdf_1 * coeff_x;

	// -- 0 y x
	interpolate_y0 = interpolate_x0 * (1 - coeff_y) + interpolate_x1 * coeff_y;




	// z y x
	// 1 0 0
	Pz = P1_z + VOXEL_SIZE;	Py = P1_y;	Px = P1_x;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_0 = voxel_block_array[voxel_index].sdf;
	// z y x
	// 1 0 1
	Pz = P1_z + VOXEL_SIZE;	Py = P1_y;	Px = P1_x + VOXEL_SIZE;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_1 = voxel_block_array[voxel_index].sdf;
	// - 1 0 x
	interpolate_x0 = sdf_0 * (1 - coeff_x) + sdf_1 * coeff_x;

	// z y x
	// 1 1 0
	Pz = P1_z + VOXEL_SIZE;	Py = P1_y + VOXEL_SIZE;	Px = P1_x;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_0 = voxel_block_array[voxel_index].sdf;
	// z y x
	// 1 1 1
	Pz = P1_z + VOXEL_SIZE;	Py = P1_y + VOXEL_SIZE;	Px = P1_x + VOXEL_SIZE;
	voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
	if (voxel_index < 0)	return false;
	if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
	sdf_1 = voxel_block_array[voxel_index].sdf;
	// - 1 1 x
	interpolate_x1 = sdf_0 * (1 - coeff_x) + sdf_1 * coeff_x;

	// --  1 y x
	interpolate_y1 = interpolate_x0 * (1 - coeff_y) + interpolate_x1 * coeff_y;

	// --- z y x
	sdf_interpolated = interpolate_y0 * (1 - coeff_z) + interpolate_y1 * coeff_z;

	return true;
}



// Compute normal vector from 8 neighbor voxel
__device__ inline bool interpolate_normal_by_sdf(float & Vx, float & Vy, float & Vz, 
												 const HashEntry * entry, const Voxel_f * voxel_block_array, 
												 My_Type::Vector3f & normal_vector)
{
	// 
	My_Type::Vector3f gradient_vector;
	//
	float sdf_000, sdf_001, sdf_010, sdf_011;
	float sdf_100, sdf_101, sdf_110, sdf_111;

	// Pre-compute coeff
	float coeff_x, coeff_y, coeff_z;
	float step = VOXEL_SIZE;
	coeff_x = (Vx + HALF_VOXEL_SIZE - floor_by_stride(Vx, step)) / step;
	coeff_y = (Vy + HALF_VOXEL_SIZE - floor_by_stride(Vy, step)) / step;
	coeff_z = (Vz + HALF_VOXEL_SIZE - floor_by_stride(Vz, step)) / step;


	// Get 8 voxel SDF value
	{
		int voxel_index;
		float Px, Py, Pz, P1_x, P1_y, P1_z;
		P1_x = Vx - HALF_VOXEL_SIZE;
		P1_y = Vy - HALF_VOXEL_SIZE;
		P1_z = Vz - HALF_VOXEL_SIZE;
		// z y x
		// 0 0 0
		Pz = P1_z;	Py = P1_y;	Px = P1_x;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_000 = voxel_block_array[voxel_index].sdf;
		// 0 0 1
		Pz = P1_z;	Py = P1_y;	Px = P1_x + VOXEL_SIZE;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_001 = voxel_block_array[voxel_index].sdf;
		// 0 1 0
		Pz = P1_z;	Py = P1_y + VOXEL_SIZE;	Px = P1_x;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_010 = voxel_block_array[voxel_index].sdf;
		// 0 1 1
		Pz = P1_z;	Py = P1_y + VOXEL_SIZE;	Px = P1_x + VOXEL_SIZE;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_011 = voxel_block_array[voxel_index].sdf;

		// 1 0 0
		Pz = P1_z + VOXEL_SIZE;	Py = P1_y;	Px = P1_x;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_100 = voxel_block_array[voxel_index].sdf;
		// 1 0 1
		Pz = P1_z + VOXEL_SIZE;	Py = P1_y;	Px = P1_x + VOXEL_SIZE;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_101 = voxel_block_array[voxel_index].sdf;
		// 1 1 0
		Pz = P1_z + VOXEL_SIZE;	Py = P1_y + VOXEL_SIZE;	Px = P1_x;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_110 = voxel_block_array[voxel_index].sdf;
		// 1 1 1
		Pz = P1_z + VOXEL_SIZE;	Py = P1_y + VOXEL_SIZE;	Px = P1_x + VOXEL_SIZE;
		voxel_index = get_voxel_index_neighbor(Px, Py, Pz, entry);
		if (voxel_index < 0)	return false;
		if (voxel_block_array[voxel_index].weight < MIN_RAYCAST_WEIGHT)	return false;
		sdf_111 = voxel_block_array[voxel_index].sdf;
	}

	//
	float gradient_00, gradient_01, gradient_10, gradient_11, gradient_0, gradient_1;
	// Gradient X
	{
		// 		_zy
		gradient_00 = sdf_001 - sdf_000;
		gradient_01 = sdf_011 - sdf_010;
		gradient_10 = sdf_101 - sdf_100;
		gradient_11 = sdf_111 - sdf_110;
		//		_z
		gradient_0 = (1 - coeff_y) * gradient_00 + coeff_y * gradient_01;
		gradient_1 = (1 - coeff_y) * gradient_10 + coeff_y * gradient_11;
		//
		normal_vector.x = (1 - coeff_z) * gradient_0 + coeff_z * gradient_1;
	}
	// Gradient Y
	{
		// 		_zx
		gradient_00 = sdf_010 - sdf_000;
		gradient_01 = sdf_011 - sdf_001;
		gradient_10 = sdf_110 - sdf_100;
		gradient_11 = sdf_111 - sdf_101;
		//		_z
		gradient_0 = (1 - coeff_x) * gradient_00 + coeff_x * gradient_01;
		gradient_1 = (1 - coeff_x) * gradient_10 + coeff_x * gradient_11;
		//
		normal_vector.y = (1 - coeff_z) * gradient_0 + coeff_z * gradient_1;
	}
	// Gradient Z
	{
		//		_yx
		gradient_00 = sdf_100 - sdf_000;
		gradient_01 = sdf_101 - sdf_001;
		gradient_10 = sdf_110 - sdf_010;
		gradient_11 = sdf_111 - sdf_011;
		//		_y
		gradient_0 = (1 - coeff_x) * gradient_00 + coeff_x * gradient_01;
		gradient_1 = (1 - coeff_x) * gradient_10 + coeff_x * gradient_11;
		//
		normal_vector.z = (1 - coeff_y) * gradient_0 + coeff_y * gradient_1;
	}

	//
	normal_vector /= normal_vector.norm();

	return true;
}



// floor_by_stride、round_by_stride、ceil_by_stride         BUG！
//     
__device__ inline float floor_by_stride(float & value, float & step)
{
	return (floorf(value / step) * step);
}

//     
__device__ inline float round_by_stride(float & value, float & step)
{
	return (roundf(value / step) * step);
}

//     
__device__ inline float ceil_by_stride(float & value, float & step)
{
	return (ceilf(value / step) * step);
}

//   hash 
__device__ inline int hash_func(int & x, int & y, int & z)
{
	return (((((unsigned int)x) * PRIME_X) ^ (((unsigned int)y) * PRIME_Y) ^ (((unsigned int)z) * PRIME_Z)) & (unsigned int)ORDERED_TABLE_MASK);
}



// GPU Warp Reduce
template<typename T>
inline __device__ void warp_reduce(volatile T * cache_T, int tid)
{
	cache_T[tid] += cache_T[tid + 32];
	cache_T[tid] += cache_T[tid + 16];
	cache_T[tid] += cache_T[tid + 8];
	cache_T[tid] += cache_T[tid + 4];
	cache_T[tid] += cache_T[tid + 2];
	cache_T[tid] += cache_T[tid + 1];
}
// Block of 256 threads Reduce
template<typename T>
inline __device__ void block_256_reduce(volatile T * cache_T, int tid)
{
	__syncthreads();
	if (tid < 128)	cache_T[tid] += cache_T[tid + 128];
	__syncthreads();
	if (tid < 64)	cache_T[tid] += cache_T[tid + 64];
	__syncthreads();
	if (tid < 32)	warp_reduce(cache_T, tid);
	__syncthreads();
}


// Block of 256 threads Reduce minimum
template<typename T>
inline __device__ void block_256_reduce_min(T * cache_T, int tid)
{
	__syncthreads();
	if (tid < 128)	cache_T[tid] = min(cache_T[tid], cache_T[tid + 128]);
	__syncthreads();
	if (tid < 64)	cache_T[tid] = min(cache_T[tid], cache_T[tid + 64]);
	__syncthreads();
	if (tid < 32)	warp_reduce_min(cache_T, tid);
	__syncthreads();

}
// GPU Warp Reduce minimum
template<typename T>
inline __device__ void warp_reduce_min(volatile T * cache_T, int tid)
{
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 32]);
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 16]);
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 8]);
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 4]);
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 2]);
	cache_T[tid] = min(cache_T[tid], cache_T[tid + 1]);
}



// Block of 256 threads Reduce maximum
template<typename T>
inline __device__ void block_256_reduce_max(T * cache_T, int tid)
{
	__syncthreads();
	if (tid < 128)	cache_T[tid] = max(cache_T[tid], cache_T[tid + 128]);
	__syncthreads();
	if (tid < 64)	cache_T[tid] = max(cache_T[tid], cache_T[tid + 64]);
	__syncthreads();
	if (tid < 32)	warp_reduce_max(cache_T, tid);
	__syncthreads();

}
// GPU Warp Reduce maximum
template<typename T>
inline __device__ void warp_reduce_max(volatile T * cache_T, int tid)
{
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 32]);
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 16]);
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 8]);
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 4]);
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 2]);
	cache_T[tid] = max(cache_T[tid], cache_T[tid + 1]);
}