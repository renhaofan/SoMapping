

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <float.h>
//
#include "Plane_KernelFunc.cuh"
#include <float.h>

// Warp Reduce
template<typename T>
inline __device__ void warp_reduce(volatile T * cache_T, int tid);
// Block of 256 threads Reduce
template<typename T>
inline __device__ void block_256_reduce(volatile T * cache_T, int tid);

// pi
//#define PI	3.1415926




// Fit plane for each patch
__global__ void fit_plane_for_cells_KernelFunc(const My_Type::Vector3f * current_points, 
											   const My_Type::Vector3f * current_normals,
											   Sensor_params sensor_params,
											   Cell_info * cell_info_mat)
{
	// Coordinate/index of cell
	int u_cell = blockIdx.x;
	int v_cell = blockIdx.y;
	int cell_index = u_cell + v_cell * gridDim.x;
	// Coordinate/index of point
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	int index = u + v * gridDim.x * blockDim.x;


	// 
	bool is_valid_cell = true;
	// Load and validate point position
	My_Type::Vector3f current_point = current_points[index];
	if (current_point.z < FLT_EPSILON)			is_valid_cell = false;
	// Load and validate point normal vector existence
	My_Type::Vector3f current_normal = current_normals[index];
	if (current_normal.norm() < FLT_EPSILON)	is_valid_cell = false;


#pragma region(Compute position of points in this cell)

	// Thread id
	int tid = threadIdx.x + blockDim.x * threadIdx.y;

	// Reduction cache buffer (shared memory)
	/*	ToDo :	How to solve "warning : __shared__ memory variable with non-empty constructor 
				or destructor (potential race between threads)" */
	__shared__ My_Type::Vector3f cache_float3[256];
	if (is_valid_cell)
	{	cache_float3[tid] = current_point;	}
	else
	{	cache_float3[tid] = 0.0f;	}
	block_256_reduce(cache_float3, tid);

	// Weight center of points in this cell
	My_Type::Vector3f weight_center =  cache_float3[0];
	__syncthreads();


	// Counter the number of valid points
	__shared__ float cache_float1[256];
	if (is_valid_cell)
	{	cache_float1[tid] = 1.0f;	}
	else
	{	cache_float1[tid] = 0.0f;	}
	block_256_reduce(cache_float1, tid);
	//
	float valid_num = cache_float1[0];
	weight_center /= valid_num;


	if (tid == 0 && is_valid_cell)
	{
		
		// Save the weight center position
		cell_info_mat[cell_index].x = weight_center.x; 
		cell_info_mat[cell_index].y = weight_center.y; 
		cell_info_mat[cell_index].z = weight_center.z; 
	}

#pragma endregion


#pragma region(Fit normal vector for each cell)

	//          
	current_point -= weight_center;
	
	// ATA = {a0, a1}	|	ATb = {b0}
	//		 {a1, a2}	|		  {b1}
	float ATA_upper[3], ATb[2];

	if (is_valid_cell)
	{
		//
		ATA_upper[0] = current_point.x * current_point.x;
		ATA_upper[1] = current_point.x * current_point.y;
		ATA_upper[2] = current_point.y * current_point.y;
		//
		ATb[0] = -current_point.x * current_point.z;
		ATb[1] = -current_point.y * current_point.z;
	}
	else
	{
		//
		ATA_upper[0] = 0.0f;
		ATA_upper[1] = 0.0f;
		ATA_upper[2] = 0.0f;

		//
		ATb[0] = 0.0f;	ATb[1] = 0.0f;
	}


	// Reduce ATA and ATb
	// ATA
	//#pragma unroll 1
	for (int i = 0; i < 3; i++)
	{
		__syncthreads();
		cache_float1[tid] = ATA_upper[i];
		// Reduce
		block_256_reduce(cache_float1, tid);
		if (tid == 0)
		{	ATA_upper[i] = cache_float1[0];	}
	}
	// ATb
	//#pragma unroll 1
	for (int i = 0; i < 2; i++)
	{
		__syncthreads();
		cache_float1[tid] = ATb[i];
		// Reduce
		block_256_reduce(cache_float1, tid);
		if (tid == 0)
		{	ATb[i] = cache_float1[0];	}
	}


	// normal vector of this cell
	__shared__ My_Type::Vector3f cell_normal_share;
	My_Type::Vector3f cell_normal;
	//    ATA x = ATb
	if (tid == 0 && is_valid_cell)
	{
		float D, D1, D2, tan_xz, tan_yz;
		// Compute Crammer
		D = ATA_upper[0] * ATA_upper[2] - ATA_upper[1] * ATA_upper[1];
		D1 = ATb[0] * ATA_upper[2] - ATb[1] * ATA_upper[1];
		D2 = ATb[1] * ATA_upper[0] - ATb[0] * ATA_upper[1];
		// compute tangent
		tan_xz = D1 / D;
		tan_yz = D2 / D;

		//      
		cell_normal.z = 1 / norm3df(tan_xz, tan_yz, 1.0f);
		cell_normal.x = tan_xz * cell_normal.z;
		cell_normal.y = tan_yz * cell_normal.z;

		//   Ray  
		My_Type::Vector3f ray;
		ray.x = ((float)u - sensor_params.sensor_cx) / sensor_params.sensor_fx;
		ray.y = ((float)v - sensor_params.sensor_cy) / sensor_params.sensor_fy;
		ray.z = 1.0f;

		if ((ray.x * cell_normal.x + ray.y * cell_normal.y + ray.z * cell_normal.z) > 0.0f)
		{
			cell_normal.x = -cell_normal.x;
			cell_normal.y = -cell_normal.y;
			cell_normal.z = -cell_normal.z;
		}

		//   Patch   
		cell_info_mat[cell_index].nx = cell_normal.x;
		cell_info_mat[cell_index].ny = cell_normal.y;
		cell_info_mat[cell_index].nz = cell_normal.z;

		//        Shared Memory        
		cell_normal_share = cell_normal;
	}



#pragma endregion


#pragma region(Counter the number of valid points)


	// Load cell normal vector for each cell
	__syncthreads();
	cell_normal = cell_normal_share;
	float prj_value = cell_normal.x * current_point.x + cell_normal.y * current_point.y + cell_normal.z * current_point.z;
	if (fabsf(prj_value) > 1e-2f)		is_valid_cell = false;


	// Reduce Patch        
	cache_float1[tid] = 0.0f;
	if (is_valid_cell)
	{	cache_float1[tid] = 1.0f;	}
	// Reduce
	block_256_reduce(cache_float1, tid);
	if (tid == 0)
	{
		if (is_valid_cell)
		{	cell_info_mat[cell_index].counter = (int)cache_float1[0];	}
		else
		{	cell_info_mat[cell_index].counter = 0;	}
	}


#pragma endregion


}
// Cpp call CUDA
void fit_plane_for_cells_CUDA(dim3 block_rect, dim3 thread_rect,
							  const My_Type::Vector3f * current_points,
							  const My_Type::Vector3f * current_normals,
							  Sensor_params sensor_params,
							  Cell_info * cell_info_mat)
{

	fit_plane_for_cells_KernelFunc << < block_rect, thread_rect >> > (current_points, current_normals, sensor_params,
																  cell_info_mat);
}



//
#define MIN_VALID_POINTS_IN_CELL	200
// 
__global__ void histogram_PxPy_KernelFunc(const Cell_info * cell_info_mat, float * hist_PxPy)
{
	// Max histogram range
	const float max_range = (HISTOGRAM_WIDTH / 2 - 1) * HISTOGRAM_STEP;

	//
	bool is_valid_cell = true;
	// Cell index
	int cell_index = threadIdx.x + blockIdx.x * blockDim.x;

	// Validate cell
	if (cell_info_mat[cell_index].counter < MIN_VALID_POINTS_IN_CELL)	is_valid_cell = false;


	// Compute histogram coordinate
	float px, py;
	//
	if (is_valid_cell)
	{
		// 
		px = 2.0f * cell_info_mat[cell_index].nx / (1.0f - cell_info_mat[cell_index].nz);
		py = 2.0f * cell_info_mat[cell_index].ny / (1.0f - cell_info_mat[cell_index].nz);
		// 
		if ((px >= max_range) || (px <= -max_range))	is_valid_cell = false;
		if ((py >= max_range) || (py <= -max_range))	is_valid_cell = false;
	}


	// 
	if (is_valid_cell)
	{
		// 
		float weighted_area = 1.0f;
		//
		float d1 = sqrtf(cell_info_mat[cell_index].nx * cell_info_mat[cell_index].nx + cell_info_mat[cell_index].ny * cell_info_mat[cell_index].ny);
		float h1 = (1.0f - cell_info_mat[cell_index].nz);
		float theta = acosf(h1 / sqrtf(h1 * h1 + d1 * d1));

		//weighted_area /= cos(theta) * cos(theta);
		weighted_area *= 4 * (cos(theta) + tan(theta / 2) * sin(theta)) / ((1 - cell_info_mat[cell_index].nz) * (1 - cell_info_mat[cell_index].nz));

		int u_hist, v_hist, index_hist;
		u_hist = (int)roundf((px + max_range) / HISTOGRAM_STEP);
		v_hist = (int)roundf((py + max_range) / HISTOGRAM_STEP);
		index_hist = u_hist + v_hist * HISTOGRAM_WIDTH;

		// 8-neighbor Gaussian sum
		atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH - 1], weighted_area / 16.0);
		atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH + 0], weighted_area / 8.0);
		atomicAdd((float *)&hist_PxPy[index_hist - HISTOGRAM_WIDTH + 1], weighted_area / 16.0);
		atomicAdd((float *)&hist_PxPy[index_hist - 1], weighted_area / 8.0);
		atomicAdd((float *)&hist_PxPy[index_hist + 0], weighted_area / 4.0);
		atomicAdd((float *)&hist_PxPy[index_hist + 1], weighted_area / 8.0);
		atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH - 1], weighted_area / 16.0);
		atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH + 0], weighted_area / 8.0);
		atomicAdd((float *)&hist_PxPy[index_hist + HISTOGRAM_WIDTH + 1], weighted_area / 16.0);
	}

}
// Cpp call CUDA 
void histogram_PxPy_CUDA(dim3 block_rect, dim3 thread_rect, 
						 Cell_info * cell_info_mat, float * hist_PxPy)
{

	histogram_PxPy_KernelFunc << < block_rect, thread_rect >> >(cell_info_mat, 
																hist_PxPy);
}



// 
__global__ void find_PxPy_peaks_KernelFunc(const float * hist_mat, Hist_normal * hist_normal, int * peak_counter)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	int index = u + v * blockDim.x * gridDim.x;

	// 
	float px = (float)(u - (HISTOGRAM_WIDTH >> 1)) * HISTOGRAM_STEP;
	float py = (float)(v - (HISTOGRAM_WIDTH >> 1)) * HISTOGRAM_STEP;
	float prj_radius = sqrtf(px * px + py * py);
	// 
	const float max_range = (HISTOGRAM_WIDTH / 2 - 1) * HISTOGRAM_STEP;
	// 
	if ((px >= max_range) || (px <= -max_range))	return;
	if ((py >= max_range) || (py <= -max_range))	return;

	// 
	float peak_value = hist_mat[index];
	if (peak_value < MIN_CELLS_OF_DIRECTION / 4.0)			return;

	// Read neigthbor value
	float neighbor_value[3][3];
	bool is_peak = true;
	//
	neighbor_value[0][0] = hist_mat[index - 1 - blockDim.x * gridDim.x];
	neighbor_value[0][1] = hist_mat[index + 0 - blockDim.x * gridDim.x];
	neighbor_value[0][2] = hist_mat[index + 1 - blockDim.x * gridDim.x];
	neighbor_value[1][0] = hist_mat[index - 1];
	neighbor_value[1][1] = hist_mat[index + 0];
	neighbor_value[1][2] = hist_mat[index + 1];
	neighbor_value[2][0] = hist_mat[index - 1 + blockDim.x * gridDim.x];
	neighbor_value[2][1] = hist_mat[index + 0 + blockDim.x * gridDim.x];
	neighbor_value[2][2] = hist_mat[index + 1 + blockDim.x * gridDim.x];
	// find maximum value
	for (int i = 0; i < 3; i++)	for (int j = 0; j < 3; j++)
	{
		if (neighbor_value[i][j] > peak_value)
		{
			is_peak = false;
			return;
		}
	}
	

	// 
	if (is_peak)
	{
		int peak_index = atomicAdd(peak_counter, 1);

		// 
		if (peak_index < MAX_HIST_NORMALS)
		{
			//  
			float nx, ny, nz, theta, radius_normal;
			theta = atanf(prj_radius / 2.0f);

			// Compute normal vector
			nz = -cos(2 * theta);
			radius_normal = sin(2 * theta);
			nx = px * radius_normal / prj_radius;
			ny = py * radius_normal / prj_radius;

			// 
			hist_normal[peak_index].nx = nx;
			hist_normal[peak_index].ny = ny;
			hist_normal[peak_index].nz = nz;
			hist_normal[peak_index].weight = peak_value;

			//printf("N[%d] = (%f, %f, %f) S = %f\r\n", peak_index, nx, ny, nz, peak_value);
		}
	}


}
// Cpp call CUDA
void find_PxPy_peaks_CUDA(dim3 block_rect, dim3 thread_rect, 
						  float * hist_mat, Hist_normal * hist_normal, int * peak_counter)
{

	find_PxPy_peaks_KernelFunc << < block_rect, thread_rect >> >(hist_mat, hist_normal, peak_counter);
}



//
#define MIN_cell_NORMAL_INNER_PRODUCT_VALUE	0.95
//
__global__ void histogram_prj_dist_KernelFunc(const Cell_info * cell_info_mat, 
											  const Hist_normal * hist_normal, 
											  float * distance_histogram)
{
	bool is_valid_cell = true;
	int cell_index = threadIdx.x + blockIdx.x * blockDim.x;

	// Validate cell 
	int valid_counter = cell_info_mat[cell_index].counter;
	if (valid_counter < MIN_VALID_POINTS_IN_CELL)	is_valid_cell = false;

	// Compute included angle between plane normal and cell normal
	My_Type::Vector3f plane_normal;
	if (is_valid_cell)
	{
		My_Type::Vector3f cell_normal;
		cell_normal.x = cell_info_mat[cell_index].nx;
		cell_normal.y = cell_info_mat[cell_index].ny;
		cell_normal.z = cell_info_mat[cell_index].nz;
		plane_normal.x = hist_normal[0].nx;
		plane_normal.y = hist_normal[0].ny;
		plane_normal.z = hist_normal[0].nz;
		float inner_product = cell_normal.x * plane_normal.x + cell_normal.y * plane_normal.y + cell_normal.z * plane_normal.z;

		if (inner_product < MIN_cell_NORMAL_INNER_PRODUCT_VALUE)	is_valid_cell = false;
	}


	// Compute project distance
	if (is_valid_cell)
	{
		My_Type::Vector3f cell_position;
		cell_position.x = cell_info_mat[cell_index].x;
		cell_position.y = cell_info_mat[cell_index].y;
		cell_position.z = cell_info_mat[cell_index].z;
		float project_distance = - plane_normal.x * cell_position.x - plane_normal.y * cell_position.y - plane_normal.z * cell_position.z;

		// Gaussian weighted 
		int dist_index = roundf(project_distance / MIN_PLANE_DISTANCE);
		if ((dist_index > (int)2) && (dist_index < (int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE) - 2))
		{
			// 0.0359	0.1093	0.2129	0.2659	0.2129	0.1093	0.0359
			atomicAdd(&distance_histogram[dist_index - 2], 0.0359);
			atomicAdd(&distance_histogram[dist_index - 1], 0.1093);
			atomicAdd(&distance_histogram[dist_index + 0], 0.2659);
			atomicAdd(&distance_histogram[dist_index + 1], 0.1093);
			atomicAdd(&distance_histogram[dist_index + 2], 0.0359);
		}
	}



}
// Cpp call CUDA
void histogram_prj_dist_CUDA(dim3 block_rect, dim3 thread_rect, 
							 const Cell_info * cell_info_mat, 
							 const Hist_normal * hist_normal, 
							 float * prj_dist_hist)
{

	histogram_prj_dist_KernelFunc << < block_rect, thread_rect >> >(cell_info_mat, hist_normal, 
																	prj_dist_hist);
}




// Find peaks on distance histogram for each direction
__global__ void find_prj_dist_peaks_KernelFunc(const float * prj_dist_hist, 
											   const Hist_normal * hist_normal,
											   int * peak_index, 
											   Plane_info * current_planes)
{
	bool is_valid_dist = true;
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// 
	float number_of_cells[3];
	number_of_cells[1] = prj_dist_hist[index];
	if (number_of_cells[1] < MIN_CELLS_OF_PLANE * 0.2659)		is_valid_dist = false;

	// 
	if (is_valid_dist)
	{
		// Read neighbor value
		number_of_cells[0] = prj_dist_hist[index - 1];
		number_of_cells[2] = prj_dist_hist[index + 1];

		// If it's a maximum. Than plane detected.
		if (number_of_cells[0] < number_of_cells[1] && number_of_cells[2] < number_of_cells[1])
		{
			int peak_index_now = atomicAdd(peak_index, 1);
			//
			current_planes[peak_index_now].nx = hist_normal[0].nx;
			current_planes[peak_index_now].ny = hist_normal[0].ny;
			current_planes[peak_index_now].nz = hist_normal[0].nz;
			current_planes[peak_index_now].d = (float)index * MIN_PLANE_DISTANCE;
			current_planes[peak_index_now].weight = prj_dist_hist[index];
			current_planes[peak_index_now].is_valid = true;
		}
	}
}
// Cpp call CUDA
void find_prj_dist_peaks_CUDA(dim3 block_rect, dim3 thread_rect, 
							  const float * prj_dist_hist, 
							  const Hist_normal * hist_normal, 
							  int * peak_index,
							  Plane_info * current_planes)
{

	find_prj_dist_peaks_KernelFunc << < block_rect, thread_rect >> >(prj_dist_hist, hist_normal, 
																	 peak_index, current_planes);
}



#pragma region (GPU K-means)
// 1. Mark plane label for each cells
__global__ void mark_plane_label_for_cells_KernelFunc(const Plane_info * current_plane, 
													  Cell_info * cell_info_mat, int plane_num)
{
	int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
	bool is_valid_cell = true;

	// Validate cell
	int valid_counter = cell_info_mat[cell_index].counter;
	if (valid_counter < MIN_VALID_POINTS_IN_CELL)	is_valid_cell = false;

	// Compute inlcuded angle
	if (is_valid_cell)
	{
		float plane_nx, plane_ny, plane_nz, nx, ny, nz;
		nx = cell_info_mat[cell_index].nx;
		ny = cell_info_mat[cell_index].ny;
		nz = cell_info_mat[cell_index].nz;
		// 
		float px, py, pz;
		px = cell_info_mat[cell_index].x;
		py = cell_info_mat[cell_index].y;
		pz = cell_info_mat[cell_index].z;

		//
		int match_plane_index = 0;
		float min_para_distance = FLT_MAX;
		for (int i = 1; i < plane_num; i++)
		{
			if (current_plane[i].is_valid == false) continue;

			plane_nx = current_plane[i].nx;
			plane_ny = current_plane[i].ny;
			plane_nz = current_plane[i].nz;
			float inner_product = nx * plane_nx + ny * plane_ny + nz * plane_nz;
			if (inner_product < 1.0f - 3 * (1.0f - MIN_cell_NORMAL_INNER_PRODUCT_VALUE))	continue;

			// Plane distance to origin
			float plane_d = current_plane[i].d;
			// Project distance
			float prj_d = -px * plane_nx - py * plane_ny - pz * plane_nz;
			// Different of distance
			float d_diff = fabsf(plane_d - prj_d);

			// 3 sigma criterion
			if (d_diff > 2 * MIN_PLANE_DISTANCE)	continue;

			// 
			if (d_diff < min_para_distance)
			{
				min_para_distance = d_diff;
				match_plane_index = i;
			}
		}

		// Match to current planes
		cell_info_mat[cell_index].plane_index = match_plane_index;
	}
	else
	{
		// No matched plane
		cell_info_mat[cell_index].plane_index = 0;
	}

}
// 2. Reset mean value
__global__ void reset_K_means_state_KernelFunc(Plane_info * current_plane)
{
	int plane_index = blockIdx.x;
	current_plane[plane_index].cell_num = 1.0;
	current_plane[plane_index].d = 0.0;
}
// 3. Sum plane parameters
__global__ void compute_mean_plane_para_KernelFunc(const Cell_info * cell_info_mat,
												   Cell_info * plane_mean_paramters, int plane_num)
{
	int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
	bool is_valid_cell = true;

	// Validate cell information
	int cell_pixel_number = cell_info_mat[cell_index].counter;
	if (cell_pixel_number < MIN_VALID_POINTS_IN_CELL)	is_valid_cell = false;
	//
	int cell_plane_index = cell_info_mat[cell_index].plane_index;
	if (cell_plane_index >= MAX_CURRENT_PLANES || cell_plane_index == 0)		is_valid_cell = false;

	//
	float px, py, pz, cell_nx, cell_ny, cell_nz;
	if (is_valid_cell)
	{
		// Load cell position 
		px = cell_info_mat[cell_index].x;
		py = cell_info_mat[cell_index].y;
		pz = cell_info_mat[cell_index].z;
	}

	// 
	for (int plane_index = 1; plane_index < plane_num; plane_index++)
	{
		__shared__ float cache_f0[256], cache_f1[256], cache_f2[256], cache_f3[256];
		int tid = threadIdx.x;

#pragma region(Reduce cell position and number-of-cell)
		//
		if (is_valid_cell && plane_index == cell_plane_index)
		{
			cache_f0[tid] = px; 
			cache_f1[tid] = py;
			cache_f2[tid] = pz;
			cache_f3[tid] = 1.0;
		}
		else
		{
			cache_f0[tid] = 0.0;
			cache_f1[tid] = 0.0;
			cache_f2[tid] = 0.0;
			cache_f3[tid] = 0.0;
		}
		// Reduce
		block_256_reduce(cache_f0, tid);
		block_256_reduce(cache_f1, tid);
		block_256_reduce(cache_f2, tid);
		block_256_reduce(cache_f3, tid);
		// Save reduce result
		if (tid == 0)
		{
			atomicAdd(&plane_mean_paramters[plane_index].x, cache_f0[0]);
			atomicAdd(&plane_mean_paramters[plane_index].y, cache_f1[0]);
			atomicAdd(&plane_mean_paramters[plane_index].z, cache_f2[0]);
			atomicAdd(&plane_mean_paramters[plane_index].counter, (int)cache_f3[0]);
		}
#pragma endregion

#pragma region(Reduce cell normal)
		// 
		if (is_valid_cell && plane_index == cell_plane_index)
		{
			cache_f0[tid] = cell_nx;
			cache_f1[tid] = cell_ny;
			cache_f2[tid] = cell_nz;
		}
		else
		{
			cache_f0[tid] = 0.0;
			cache_f1[tid] = 0.0;
			cache_f2[tid] = 0.0;
		}
		// Reduce
		block_256_reduce(cache_f0, tid);
		block_256_reduce(cache_f1, tid);
		block_256_reduce(cache_f2, tid);
		// 
		if (tid == 0)
		{
			atomicAdd(&plane_mean_paramters[plane_index].nx, cache_f0[0]);
			atomicAdd(&plane_mean_paramters[plane_index].ny, cache_f1[0]);
			atomicAdd(&plane_mean_paramters[plane_index].nz, cache_f2[0]);
		}
#pragma endregion
	}
}
// 3. Compute mean parameter of planes
__global__ void compute_plane_mean_para_KernelFunc(Cell_info * plane_mean_paramters, int plane_num)
{
	int plane_index = blockIdx.x;
	// Validate
	int cell_pixel_number = plane_mean_paramters[plane_index].counter;
	if (cell_pixel_number == 0)	return;

	// Compute mean value
	plane_mean_paramters[plane_index].x /= (float)cell_pixel_number;
	plane_mean_paramters[plane_index].y /= (float)cell_pixel_number;
	plane_mean_paramters[plane_index].z /= (float)cell_pixel_number;
	plane_mean_paramters[plane_index].nx /= (float)cell_pixel_number;
	plane_mean_paramters[plane_index].ny /= (float)cell_pixel_number;
	plane_mean_paramters[plane_index].nz /= (float)cell_pixel_number;
}
// 4. Re-compute plane normal vector (step 1)
__global__ void recompute_plane_normal_step1_KernelFunc(const Cell_info * cell_info_mat, 
														const Cell_info * plane_mean_paramters, 
														float * ATA_upper_buffer, float * ATb_buffer, 
														Plane_info * current_plane,	int plane_num)
{
	int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
	bool is_valid_cell = true;
	// Validate cell
	int cell_pixel_number = cell_info_mat[cell_index].counter;
	if (cell_pixel_number < MIN_VALID_POINTS_IN_CELL)		is_valid_cell = false;
	// Validate plane index
	int cell_plane_index = cell_info_mat[cell_index].plane_index;
	if (cell_plane_index >= MAX_CURRENT_PLANES || cell_plane_index == 0)	is_valid_cell = false;

	//
	for (int plane_index = 1; plane_index < plane_num; plane_index++)
	{
		float px, py, pz;
		if (is_valid_cell && plane_index == cell_plane_index)
		{
			// Load position
			px = cell_info_mat[cell_index].x - plane_mean_paramters[plane_index].x;
			py = cell_info_mat[cell_index].y - plane_mean_paramters[plane_index].y;
			pz = cell_info_mat[cell_index].z - plane_mean_paramters[plane_index].z;
		}

		//
		__shared__ float cache_f[256];
		int tid = threadIdx.x;

		// ATA = {a0, a1}	|	ATb = {b0}
		//		 {a1, a2}	|		  {b1}
		float ATA_upper[3], ATb[2];
		// Reduce ATA and ATb
		if (is_valid_cell && plane_index == cell_plane_index)
		{
			ATA_upper[0] = px * px;
			ATA_upper[1] = px * py;
			ATA_upper[2] = py * py;
			ATb[0] = -px * pz;	ATb[1] = -py * pz;
		}
		else
		{
			ATA_upper[0] = 0.0f;
			ATA_upper[1] = 0.0f;
			ATA_upper[2] = 0.0f;
			ATb[0] = 0.0f;	ATb[1] = 0.0f;
		}
		// Reduce Hessian
		for (int i = 0; i < 3; i++)
		{
			__syncthreads();
			cache_f[tid] = ATA_upper[i];
			// Reduce
			block_256_reduce(cache_f, tid);
			if (tid == 0)	atomicAdd(&ATA_upper_buffer[plane_index * 3 + i], cache_f[0]);
		}
		// Reduce Nabla
		for (int i = 0; i < 2; i++)
		{
			__syncthreads();
			cache_f[tid] = ATb[i];
			block_256_reduce(cache_f, tid);
			if (tid == 0)	atomicAdd(&ATb_buffer[plane_index * 2 + i], cache_f[0]);
		}


		// Count valid cells
		if (is_valid_cell && plane_index == cell_plane_index)
		{
			cache_f[tid] = 1.0;
		}
		else
		{
			cache_f[tid] = 0.0;
		}
		block_256_reduce(cache_f, tid);
		if (tid == 0)	atomicAdd(&current_plane[plane_index].cell_num, cache_f[0]);
	}

}
// 4. Re-compute plane normal vector (step 2)
__global__ void recompute_plane_normal_step2_KernelFunc(Plane_info * current_plane, Cell_info * plane_mean_paramters, 
														float * ATA_upper_buffer, float * ATb_buffer)
{
	int plane_index = blockIdx.x;
	// Validate
	int cell_num = (int)current_plane[plane_index].cell_num;
	if (cell_num <= MIN_CELLS_OF_PLANE)
	{
		current_plane[plane_index].is_valid = false;
		return;
	}

	// 
	float ATA_upper[3], ATb[2];
	ATA_upper[0] = ATA_upper_buffer[plane_index * 3 + 0];
	ATA_upper[1] = ATA_upper_buffer[plane_index * 3 + 1];
	ATA_upper[2] = ATA_upper_buffer[plane_index * 3 + 2];
	ATb[0] = ATb_buffer[plane_index * 2 + 0];
	ATb[1] = ATb_buffer[plane_index * 2 + 1];

	// Cramer solver
	float D, D1, D2, tan_xz, tan_yz;
	D = ATA_upper[0] * ATA_upper[2] - ATA_upper[1] * ATA_upper[1];
	D1 = ATb[0] * ATA_upper[2] - ATb[1] * ATA_upper[1];
	D2 = ATb[1] * ATA_upper[0] - ATb[0] * ATA_upper[1];
	// compte tangent theta
	tan_xz = D1 / D;
	tan_yz = D2 / D;

	// 
	float nx, ny, nz;
	// Validate hessian
	if (D > FLT_EPSILON)
	{
		nz = 1 / norm3df(tan_xz, tan_yz, 1.0f);
		nx = tan_xz * nz;
		ny = tan_yz * nz;
	}
	else
	{
		nx = plane_mean_paramters[plane_index].nx;
		ny = plane_mean_paramters[plane_index].ny;
		nz = plane_mean_paramters[plane_index].nz;
	}

	// Check direction of normal vector (normal vector are difined as the vector point to camera)
	float ray_x, ray_y, ray_z;
	ray_x = plane_mean_paramters[plane_index].x;
	ray_y = plane_mean_paramters[plane_index].y;
	ray_z = plane_mean_paramters[plane_index].z;
	if ((ray_x * nx + ray_y * ny + ray_z * nz) > 0.0f)
	{	nx = -nx;	ny = -ny;	nz = -nz;	}

	// Update current plane direction
	current_plane[plane_index].nx = nx;
	current_plane[plane_index].ny = ny;
	current_plane[plane_index].nz = nz;
}
// 5. Re-compute plane distance to origin
__global__ void recompute_plane_distance_step1_KernelFunc(const Cell_info * cell_info_mat,
														  Plane_info * current_plane, int plane_num)
{
	int cell_index = threadIdx.x + blockDim.x * blockIdx.x;
	bool is_valid_cell = true;

	// Validate cell 
	int valid_counter = cell_info_mat[cell_index].counter;
	if (valid_counter < MIN_VALID_POINTS_IN_CELL)		is_valid_cell = false;
	// Validate plane
	int cell_plane_index = cell_info_mat[cell_index].plane_index;
	if (cell_plane_index > MAX_CURRENT_PLANES || cell_plane_index == 0)	is_valid_cell = false;

	//
	__shared__ float cache_f[256];
	int tid = threadIdx.x;
	//
	float plane_nx, plane_ny, plane_nz, prj_distance;
	for (int plane_index = 1; plane_index < plane_num; plane_index++)
	{
		// Load plane normal direction
		plane_nx = current_plane[plane_index].nx;
		plane_ny = current_plane[plane_index].ny;
		plane_nz = current_plane[plane_index].nz;
		// Compute projection distance to origin
		if (is_valid_cell && cell_plane_index == plane_index)
		{
			float cell_x, cell_y, cell_z;
			cell_x = cell_info_mat[cell_index].x;
			cell_y = cell_info_mat[cell_index].y;
			cell_z = cell_info_mat[cell_index].z;

			prj_distance = -plane_nx * cell_x - plane_ny * cell_y - plane_nz * cell_z;
		}
		else
		{
			prj_distance = 0.0;
		}

		// Reduce plane distance
		cache_f[tid] = prj_distance;
		block_256_reduce(cache_f, tid);
		if (tid == 0)	atomicAdd(&current_plane[plane_index].d, cache_f[0]);
	}
}
// （2）
__global__ void recompute_plane_distance_step2_KernelFunc(Plane_info * current_plane)
{
	//
	int plane_index = blockIdx.x;
	current_plane[plane_index].d /= current_plane[plane_index].cell_num;
}
// Cpp call CUDA
void K_mean_iterate_CUDA(dim3 block_rect, dim3 thread_rect, 
						 Plane_info * current_plane, 
						 Cell_info * cell_info_mat, 
						 Cell_info * plane_mean_paramters, 
						 float * ATA_upper_buffer, float * ATb_buffer, int plane_num)
{

	// 1. Mark plane label for each cell
	mark_plane_label_for_cells_KernelFunc << < block_rect, thread_rect >> >(current_plane, cell_info_mat, plane_num);
	
	// 2. Reset plane parameters 
	reset_K_means_state_KernelFunc << < plane_num, 1 >> >(current_plane);

	// 3. Compute mean value of each planes (plane label are marked in step 1.)
	compute_mean_plane_para_KernelFunc << < block_rect, thread_rect >> >(cell_info_mat, plane_mean_paramters, plane_num);
	compute_plane_mean_para_KernelFunc << < plane_num, 1 >> >(plane_mean_paramters, plane_num);

	// 4. Recompute current plane direction
	recompute_plane_normal_step1_KernelFunc << <block_rect, thread_rect >> >(cell_info_mat, plane_mean_paramters, 
																			 ATA_upper_buffer, ATb_buffer, 
																			 current_plane, plane_num);
	recompute_plane_normal_step2_KernelFunc << <plane_num, 1 >> >(current_plane, plane_mean_paramters, 
																  ATA_upper_buffer, ATb_buffer);

	// 5. Recompute current plane distance to origin
	recompute_plane_distance_step1_KernelFunc << <block_rect, thread_rect >> >(cell_info_mat, 
																			   current_plane, plane_num);
	recompute_plane_distance_step2_KernelFunc << <plane_num, 1 >> > (current_plane);
}
#pragma endregion



//
__global__ void label_current_planes_KernelFunc(const Cell_info * cell_info_mat, 
												int * current_plane_labels)
{
	// Coordinate/index of cell
	int u_cell = blockIdx.x;
	int v_cell = blockIdx.y;
	int cell_index = u_cell + v_cell * gridDim.x;
	// Coordinate/index of pixel
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	int index = u + v * gridDim.x * blockDim.x;

	//
	int plane_label = cell_info_mat[cell_index].plane_index;
	current_plane_labels[index] = plane_label;
}
//
void label_current_planes_CUDA(dim3 block_rect, dim3 thread_rect,
							   const Cell_info * cell_info_mat,
							   int * current_plane_labels)
{

	label_current_planes_KernelFunc << <block_rect, thread_rect >> >(cell_info_mat, 
																	 current_plane_labels);
}



//
__global__ void count_planar_pixel_number_KernelFunc(const int * plane_labels,
													 Plane_info * plane_list, 
													 int plane_counter)
{
	int u = threadIdx.x + blockDim.x * blockIdx.x;
	int v = threadIdx.y + blockDim.y * blockIdx.y;
	int index = u + v * gridDim.x * blockDim.x;

	int plane_label = plane_labels[index];
	//
	__shared__ int counter_cache[256];
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	// Count each plane
	for (int plane_index = 0; plane_index < plane_counter; plane_index++)
	{

		//
		if (plane_index == plane_label)
		{	counter_cache[tid] = 1;	}
		else
		{	counter_cache[tid] = 0;	}

		//
		block_256_reduce(counter_cache, tid);
		if (tid == 0)
		{
			atomicAdd(&plane_list[plane_index].pixel_number, counter_cache[0]);
		}
	}
}
//
void count_planar_pixel_number_CUDA(dim3 block_rect, dim3 thread_rect, 
									const int * plane_labels,
									Plane_info * plane_list,
									int plane_counter)
{

	count_planar_pixel_number_KernelFunc << <block_rect, thread_rect >> >(plane_labels,
																		  plane_list, plane_counter);
}


//
__global__ void count_overlap_pxiel_number_KernelFunc(const int * current_plane_labels,
													  const int * model_plane_labels,
													  int current_plane_counter,
													  int * relative_matrix)
{
	int u = threadIdx.x + blockDim.x * blockIdx.x;
	int v = threadIdx.y + blockDim.y * blockIdx.y;
	int index = u + v * gridDim.x * blockDim.x;

	//
	int current_label = current_plane_labels[index];
	int model_label = model_plane_labels[index];

	//
	__shared__ int model_plane_counter_cache[MAX_MODEL_PLANES];
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	for (int plane_id = 0; plane_id < current_plane_counter; plane_id++)
	{
		for (int i = 0; i < MAX_MODEL_PLANES / 256; i++)
		{	model_plane_counter_cache[tid + 256 * i] = 0;	}
		__syncthreads();

		if (plane_id == current_label)
		{	atomicAdd(&model_plane_counter_cache[model_label], 1);	}
		__syncthreads();

		//
		for (int i = 0; i < MAX_MODEL_PLANES / 256; i++)
		{	relative_matrix[tid + 256 * i] = model_plane_counter_cache[tid + 256 * i];	}
		__syncthreads();

	}
}
//
void count_overlap_pxiel_number_CUDA(dim3 block_rect, dim3 thread_rect, 
									 const int * current_plane_labels,
									 const int * model_plane_labels,
									 int current_plane_counter,
									 int * relative_matrix)
{

	count_overlap_pxiel_number_KernelFunc << <block_rect, thread_rect >> >(current_plane_labels, model_plane_labels, current_plane_counter,
																		   relative_matrix);
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

