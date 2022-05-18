


// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>


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
//
// Block of 512 threads Reduce
template<typename T>
inline __device__ void block_512_reduce(volatile T * cache_T, int tid)
{
	__syncthreads();
	if (tid < 256)	cache_T[tid] += cache_T[tid + 256];
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