//! Note this function must not inlcuded by header files




// voxel        
#include "voxel_definition.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

//
#include "math.h"
#include "OurLib/reduction_KernelFunc.cuh"


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


// 
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
	if ((entry[hash_value].position[0] == block_pos[0]) && 
		(entry[hash_value].position[1] == block_pos[1]) && 
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
			if ((entry[excess_offset].position[0] == block_pos[0]) && 
				(entry[excess_offset].position[1] == block_pos[1]) && 
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


__device__ inline bool get_sdf_interpolated(float & Vx, float & Vy, float & Vz,
											const HashEntry * entry, const Voxel_f * voxel_block_array,
											float & sdf_interpolated)
{
	int voxel_index;
	float sdf_0, sdf_1, interpolate_x0, interpolate_x1, interpolate_y0, interpolate_y1;

	//     
	float step = VOXEL_SIZE;
	float coeff_x, coeff_y, coeff_z;
	float temp_f[3];
	temp_f[0] = Vx + HALF_VOXEL_SIZE;
	temp_f[1] = Vy + HALF_VOXEL_SIZE;
	temp_f[2] = Vz + HALF_VOXEL_SIZE;
	coeff_x = (temp_f[0] - floor_by_stride(temp_f[0], step)) / step;
	coeff_y = (temp_f[1] - floor_by_stride(temp_f[1], step)) / step;
	coeff_z = (temp_f[2] - floor_by_stride(temp_f[2], step)) / step;


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


