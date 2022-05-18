


//
#include "Map_engine/Plane_map_KernelFunc.cuh"
#include "Map_engine/Voxel_device_interface.cuh"
#include "Map_engine/Plane_pixel_device_interface.cuh"
#include <float.h>




//
__device__ inline void min_max_2D_bounding_box(const My_Type::Vector2f vec, 
											   My_Type::Vector2f * bounding_box)
{
	bounding_box[0].x = min(vec.x, bounding_box[0].x);
	bounding_box[0].y = min(vec.y, bounding_box[0].y);
	bounding_box[1].x = max(vec.x, bounding_box[1].x);
	bounding_box[1].y = max(vec.y, bounding_box[1].y);
}
//
__device__ inline void compute_2D_bounding_box(const My_Type::Vector3f & vertex,
											   Plane_coordinate plane_coordinate, 
											   My_Type::Vector2f * bounding_box)
{
	My_Type::Vector3f temp_vertex, plane_normal;

	//
	const float block_width = VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	My_Type::Vector2f vertex_on_plane;

	// x-0 y-0 z-0 
	temp_vertex = vertex + My_Type::Vector3f(0.0f, 0.0f, 0.0f);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-1 y-0 z-0 
	temp_vertex = vertex + My_Type::Vector3f(block_width, 0.0f, 0.0f);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-0 y-1 z-0 
	temp_vertex = vertex + My_Type::Vector3f(0.0f, block_width, 0.0f);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-1 y-1 z-0 
	temp_vertex = vertex + My_Type::Vector3f(block_width, block_width, 0.0f);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);

	// x-0 y-0 z-1 
	temp_vertex = vertex + My_Type::Vector3f(0.0f, 0.0f, block_width);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-1 y-0 z-1 
	temp_vertex = vertex + My_Type::Vector3f(block_width, 0.0f, block_width);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-0 y-1 z-1 
	temp_vertex = vertex + My_Type::Vector3f(0.0f, block_width, block_width);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
	// x-1 y-1 z-1 
	temp_vertex = vertex + My_Type::Vector3f(block_width, block_width, block_width);
	vertex_on_plane.x = temp_vertex.dot(plane_coordinate.x_vec);
	vertex_on_plane.y = temp_vertex.dot(plane_coordinate.y_vec);
	min_max_2D_bounding_box(vertex_on_plane, bounding_box);
}
//
__global__ void build_allocate_flag_KernelFunc(const HashEntry * allocated_entries,
											   const Voxel_f * voxel_block_array,
											   const PlaneHashEntry * plane_entries,
											   Plane_coordinate plane_coordinate,
											   int model_plane_id,
											   bool * need_allocate,
											   My_Type::Vector2i * allocate_pixel_block_pos)
{
	//
	bool is_valid_entry = true;
	// 
	int voxel_entry_id = blockIdx.x;

	int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
	// Validate plane label
	__shared__ bool is_matched_plane;
	__shared__ int planar_counter_cache[512];
	is_matched_plane = false;
	__syncthreads();
	if (is_valid_entry)
	{
		int voxel_index = allocated_entries[voxel_entry_id].ptr;
		voxel_index += tid;
		if (fabsf(voxel_block_array[voxel_index].sdf) < 0.3 &&
			voxel_block_array[voxel_index].plane_index == model_plane_id)
		{	is_matched_plane = true;	}

		__syncthreads();
		if (!is_matched_plane)	is_valid_entry = false;
	}

	//
	My_Type::Vector3f voxel_block_vertex(0.0f);
	float block_width = VOXEL_BLOCK_WDITH * VOXEL_SIZE;
	if (is_valid_entry)
	{
		voxel_block_vertex.x = allocated_entries[voxel_entry_id].position[0] * block_width;
		voxel_block_vertex.y = allocated_entries[voxel_entry_id].position[1] * block_width;
		voxel_block_vertex.z = allocated_entries[voxel_entry_id].position[2] * block_width;
	}

	// The 2D bounding box
	My_Type::Vector2f pixel_bounding_box[2];
	if (tid == 0 && is_valid_entry)
	{
		My_Type::Vector2f vertex_on_plane;
		vertex_on_plane.x = voxel_block_vertex.dot(plane_coordinate.x_vec);
		vertex_on_plane.y = voxel_block_vertex.dot(plane_coordinate.y_vec);

		pixel_bounding_box[0] = My_Type::Vector2f(+FLT_MAX);
		pixel_bounding_box[1] = My_Type::Vector2f(-FLT_MAX);
		compute_2D_bounding_box(voxel_block_vertex, plane_coordinate, pixel_bounding_box);
		//
		float pixel_block_width = PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_SIZE;
		int x_start = floor_by_stride(pixel_bounding_box[0].x, pixel_block_width) / pixel_block_width + FLT_EPSILON;
		int x_end = floor_by_stride(pixel_bounding_box[1].x, pixel_block_width) / pixel_block_width + FLT_EPSILON;
		int y_start = floor_by_stride(pixel_bounding_box[0].y, pixel_block_width) / pixel_block_width + FLT_EPSILON;
		int y_end = floor_by_stride(pixel_bounding_box[1].y, pixel_block_width) / pixel_block_width + FLT_EPSILON;
		//
		for (int pixel_block_x = x_start; pixel_block_x <= x_end; pixel_block_x++)
			for (int pixel_block_y = y_start; pixel_block_y <= y_end; pixel_block_y++)
		{
			int hash_value = plane_hash_func(pixel_block_x, pixel_block_y);

			// Check allocated
			bool is_allocated = false;
			int entry_index = hash_value;
			do{
				PlaneHashEntry temp_entry = plane_entries[entry_index];
				if ((temp_entry.position[0] == pixel_block_x) && 
					(temp_entry.position[1] == pixel_block_y))
				{
					is_allocated = true;
					break;
				}

				//
				entry_index = temp_entry.offset;
			} while (entry_index >= 0);

			//
			if (!is_allocated)
			{
				need_allocate[hash_value] = true;
				allocate_pixel_block_pos[hash_value] = My_Type::Vector2i(pixel_block_x, pixel_block_y);
			}
		}
	}
}
//
void build_allocate_flag_CUDA(dim3 block_rect, dim3 thread_rect,
							  const HashEntry * allocated_entries,
							  const Voxel_f * voxel_block_array,
							  const PlaneHashEntry * plane_entries,
							  Plane_coordinate plane_coordinate,
							  int model_plane_id,
							  bool * need_allocate,
							  My_Type::Vector2i * allocate_pixel_block_pos)
{

	build_allocate_flag_KernelFunc << < block_rect, thread_rect >> >(allocated_entries, voxel_block_array, plane_entries,
																	 plane_coordinate, model_plane_id,
																	 need_allocate, allocate_pixel_block_pos);
}




//
__global__ void allocate_plane_blocks_KernelFunc(bool * need_allocate,
												 const My_Type::Vector2i * allocate_pixel_block_pos,
												 PlaneHashEntry * plane_entries, 
												 int * excess_counter,
												 int * number_of_blocks)
{
	int hash_value = threadIdx.x + blockIdx.x * blockDim.x;
	bool entry_need_allocate = need_allocate[hash_value];
	//
	if (entry_need_allocate)
	{
		//
		PlaneHashEntry plane_entry = plane_entries[hash_value];
		//
		int entry_index = hash_value;
		if (plane_entry.ptr >= 0)
		{
			int excess_offset = plane_entry.offset;
			int pre_offset = hash_value;
			while (excess_offset >= 0)
			{
				pre_offset = excess_offset;
				excess_offset = plane_entries[pre_offset].offset;
			}
			//
			excess_offset = atomicAdd(excess_counter, 1) + ORDERED_PLANE_TABLE_LENGTH;
			plane_entries[pre_offset].offset = excess_offset;
			//
			if (excess_offset >= (ORDERED_PLANE_TABLE_LENGTH + EXCESS_PLANE_TABLE_LENGTH))	return;
			entry_index = excess_offset;
		}

		// Allocate
		plane_entries[entry_index].position[0] = allocate_pixel_block_pos[hash_value].x;
		plane_entries[entry_index].position[1] = allocate_pixel_block_pos[hash_value].y;
		plane_entries[entry_index].ptr = (int)PLANE_PIXEL_BLOCK_SIZE * atomicAdd(number_of_blocks, 1);
	}

}
//
void allocate_plane_blocks_CUDA(dim3 block_rect, dim3 thread_rect,
								bool * need_allocate,
								const My_Type::Vector2i * allocate_pixel_block_pos,
								PlaneHashEntry * plane_entries,
								int * excess_counter,
								int * number_of_blocks)
{

	allocate_plane_blocks_KernelFunc << <block_rect, thread_rect >> >(need_allocate, allocate_pixel_block_pos,
																	  plane_entries, excess_counter, number_of_blocks);
}



//
__global__ void find_allocated_planar_entries_KernelFunc(const PlaneHashEntry * entries,
														 int * index_array, int * allocated_counter)
{
	bool is_allocated = true;
	int entry_index = blockIdx.x * blockDim.x + threadIdx.x;
	//
	if (entries[entry_index].ptr >= 0)	index_array[atomicAdd(allocated_counter, 1)] = entry_index;
}
//
void find_allocated_planar_entries_CUDA(dim3 block_rect, dim3 thread_rect,
										const PlaneHashEntry * entries,
										int * index_array, int * allocated_counter)
{

	find_allocated_planar_entries_KernelFunc << <block_rect, thread_rect >> >(entries,
																			  index_array, allocated_counter);
}


//
__global__ void fusion_sdf_to_plane_KernelFunc(const HashEntry * voxel_entries,
											   const Voxel_f * voxel_block_array,
											   const int * entries_index_list,
											   Plane_info model_plane,
											   Plane_coordinate plane_coordinate,
											   PlaneHashEntry * plane_entries,
											   Plane_pixel * plane_pixel_array)
{
	//
	bool is_valid_pixel = true;
	// Entry id
	int entry_id = entries_index_list[blockIdx.x];

	//
	int pixel_block_index = plane_entries[entry_id].ptr;
	if (pixel_block_index < 0)	is_valid_pixel = false;

	
	//
	float sdf;
	My_Type::Vector3f pixel_point;
	if (is_valid_pixel)
	{
		//
		float px = plane_entries[entry_id].position[0] * PLANE_PIXEL_BLOCK_WIDTH_M + threadIdx.x * PLANE_PIXEL_SIZE + HALF_PLANE_PIXEL_SIZE;
		float py = plane_entries[entry_id].position[1] * PLANE_PIXEL_BLOCK_WIDTH_M + threadIdx.y * PLANE_PIXEL_SIZE + HALF_PLANE_PIXEL_SIZE;
		float distance = - model_plane.d;

		//
		pixel_point.x = px * plane_coordinate.x_vec.x + py * plane_coordinate.y_vec.x + distance * plane_coordinate.z_vec.x;
		pixel_point.y = px * plane_coordinate.x_vec.y + py * plane_coordinate.y_vec.y + distance * plane_coordinate.z_vec.y;
		pixel_point.z = px * plane_coordinate.x_vec.z + py * plane_coordinate.y_vec.z + distance * plane_coordinate.z_vec.z;

		//
		int voxel_index = get_voxel_index_neighbor(pixel_point.x, pixel_point.y, pixel_point.z, voxel_entries);
		if (voxel_index >= 0)
		{
			if (voxel_block_array[voxel_index].weight <= MIN_RAYCAST_WEIGHT)			is_valid_pixel = false;
			if (voxel_block_array[voxel_index].plane_index != model_plane.plane_index)	is_valid_pixel = false;
			//
			is_valid_pixel |= get_sdf_interpolated(pixel_point.x, pixel_point.y, pixel_point.z, voxel_entries, voxel_block_array, sdf);
			if (fabsf(sdf) > 0.99)	is_valid_pixel = false;
		}
		else
		{
			is_valid_pixel = false;

		}

	}

	// Validate
	float diff_length = 0.0f;
	bool find_small_sdf = false;
	if (is_valid_pixel)
	{
		float max_sdf_value = sdf;
		float diff_ratio = 0.2;
		float last_diff_length;
		My_Type::Vector3f temp_point(pixel_point);
		for (int i = 0; i < 8; i++)
		{
			diff_length -= diff_ratio * sdf * TRUNCATED_BAND;
			temp_point = pixel_point + diff_length * plane_coordinate.z_vec;
			is_valid_pixel |= get_sdf_interpolated(temp_point.x, temp_point.y, temp_point.z, voxel_entries, voxel_block_array, sdf);
			//
			if (fabsf(sdf) > 0.99 || !is_valid_pixel)
			{ 
				is_valid_pixel = true;
				diff_length = last_diff_length;
				diff_ratio *= 0.5;
			}
			else
			{
				last_diff_length = diff_length;
				diff_ratio *= 2;
			}
			if (fabsf(sdf) < 0.1)	
			{ 
				find_small_sdf = true;
				break;	 
			}
		}
		//if (fabsf(sdf) > 0.1)	is_valid_pixel = false;
		const float max_diff = 0.10;
		if (!find_small_sdf || diff_length > max_diff)
		{
			is_valid_pixel |= get_sdf_interpolated(pixel_point.x, pixel_point.y, pixel_point.z, voxel_entries, voxel_block_array, sdf);
			diff_length = 0.0f;
			is_valid_pixel = false;
		}
	}

	//
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int pixel_index = pixel_block_index + tid;
	if (is_valid_pixel)
	{
		plane_pixel_array[pixel_index].diff = diff_length - sdf * TRUNCATED_BAND;
		//plane_pixel_array[pixel_index].diff = diff_length ;
		plane_pixel_array[pixel_index].plane_label = model_plane.plane_index;
	}
	else
	{
		plane_pixel_array[pixel_index].diff = FLT_MAX;
	}
	
}
//
void fusion_sdf_to_plane_CUDA(dim3 block_rect, dim3 thread_rect,
							  const HashEntry * allocated_entries,
							  const Voxel_f * voxel_block_array,
							  const int * entries_index_list,
							  Plane_info model_plane,
							  Plane_coordinate plane_coordinate,
							  PlaneHashEntry * plane_entries,
							  Plane_pixel * plane_pixel_array)
{

	fusion_sdf_to_plane_KernelFunc << <block_rect, thread_rect >> >(allocated_entries, voxel_block_array, entries_index_list,
																	model_plane, plane_coordinate,
																	plane_entries, plane_pixel_array);
}




//
__global__ void generate_block_vertex_KernelFunc(const PlaneHashEntry * entries,
												 Plane_coordinate plane_coordinate,
												 Plane_info model_plane,
												 My_Type::Vector3f * block_vertexs,
												 int * number_of_blocks)
{
	bool is_allocated = true;
	int entry_index = blockIdx.x * blockDim.x + threadIdx.x;

	// Load entry
	PlaneHashEntry temp_entry = entries[entry_index];
	if (temp_entry.ptr < 0)	is_allocated = false;

	// Mark allocated entries
	__shared__ My_Type::Vector3f vertex_buffer[256 * 8];
	__shared__ int cache_block_number;
	cache_block_number = 0;
	__syncthreads();
	int tid = threadIdx.x;
	if (is_allocated)
	{
		int buffer_index = 8 * atomicAdd(&cache_block_number, 1);

		//
		float pixel_block_width = PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_SIZE;
		My_Type::Vector3f temp_vertex[4];
		// 00
		temp_vertex[0] = pixel_block_width * (float)temp_entry.position[0] * plane_coordinate.x_vec +
						 pixel_block_width * (float)temp_entry.position[1] * plane_coordinate.y_vec +
						 (float)(-model_plane.d) * plane_coordinate.z_vec;
		//// 10
		//temp_vertex[1] = temp_vertex[0] + pixel_block_width * plane_coordinate.y_vec;
		//// 11
		//temp_vertex[2] = temp_vertex[0] + pixel_block_width * (plane_coordinate.x_vec + plane_coordinate.y_vec);
		//// 01
		//temp_vertex[3] = temp_vertex[0] + pixel_block_width * plane_coordinate.x_vec;
		// 10
		temp_vertex[1] = temp_vertex[0] + 0.8 * pixel_block_width * plane_coordinate.y_vec;
		// 11
		temp_vertex[2] = temp_vertex[0] + 0.8 * pixel_block_width * (plane_coordinate.x_vec + plane_coordinate.y_vec);
		// 01
		temp_vertex[3] = temp_vertex[0] + 0.8 * pixel_block_width * plane_coordinate.x_vec;

		//
		vertex_buffer[buffer_index + 0] = temp_vertex[0];	vertex_buffer[buffer_index + 1] = temp_vertex[1];
		vertex_buffer[buffer_index + 2] = temp_vertex[1];	vertex_buffer[buffer_index + 3] = temp_vertex[2];
		vertex_buffer[buffer_index + 4] = temp_vertex[2];	vertex_buffer[buffer_index + 5] = temp_vertex[3];
		vertex_buffer[buffer_index + 6] = temp_vertex[3];	vertex_buffer[buffer_index + 7] = temp_vertex[0];
	}


	// Atomic get allocated_array offset
	__syncthreads();
	__shared__ int number_of_written_block;
	if (tid == 0)
	{
		if (cache_block_number)
		{
			//printf("cache_block_number = %d\n", cache_block_number);
			number_of_written_block = atomicAdd(number_of_blocks, cache_block_number);
		}
	}

	// Save allocated entries
	__syncthreads();
	if (tid < cache_block_number)
	{
		int array_offset = number_of_written_block * 8;
		block_vertexs[array_offset + tid * 8 + 0] = vertex_buffer[tid * 8 + 0];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 1] = vertex_buffer[tid * 8 + 1];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 2] = vertex_buffer[tid * 8 + 2];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 3] = vertex_buffer[tid * 8 + 3];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 4] = vertex_buffer[tid * 8 + 4];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 5] = vertex_buffer[tid * 8 + 5];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 6] = vertex_buffer[tid * 8 + 6];		__threadfence_block();
		block_vertexs[array_offset + tid * 8 + 7] = vertex_buffer[tid * 8 + 7];		__threadfence_block();
	}
}
//
void generate_block_vertex_CUDA(dim3 block_rect, dim3 thread_rect,
								const PlaneHashEntry * entries,
								Plane_coordinate plane_coordinate,
								Plane_info model_plane,
								My_Type::Vector3f * block_vertexs,
								int * number_of_blocks)
{

	generate_block_vertex_KernelFunc << <block_rect, thread_rect >> >(entries, plane_coordinate, model_plane,
																	  block_vertexs, number_of_blocks);
}


