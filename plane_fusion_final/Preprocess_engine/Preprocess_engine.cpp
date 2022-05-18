#include "Preprocess_engine.h"


// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

// CUDA functions
#include "Preprocess_KernelFunc.cuh"


#pragma region(CUDA memory operation for hierarchy image)
// CUDA memory allocation for hierarchy image
template <typename T>
void allocate_CUDA_memory_for_hierarchy(Hierarchy_image<T> & hierarcgy_image)
{
	for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers; layer_id++)
	{
		checkCudaErrors(cudaMalloc((void **)&(hierarcgy_image.data_ptrs[layer_id]),
			hierarcgy_image.size[layer_id].width * hierarcgy_image.size[layer_id].height * sizeof(T)));
		checkCudaErrors(cudaMemset(hierarcgy_image.data_ptrs[layer_id], 0x00,
			hierarcgy_image.size[layer_id].width * hierarcgy_image.size[layer_id].height * sizeof(T)));
	}
}
// CUDA memory free for hierarchy image
template <typename T>
void release_CUDA_memory_for_hierarchy(Hierarchy_image<T> & hierarcgy_image)
{
	for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers; layer_id++)
		checkCudaErrors(cudaFree(hierarcgy_image.data_ptrs[layer_id]));
}

#pragma endregion


#pragma region(HOST memory operation for hierarchy image)
// Host memory allocation for hierarchy image
template <typename T>
void allocate_host_memory_for_hierarchy(Hierarchy_image<T> & hierarcgy_image)
{
	for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers; layer_id++)
	{
		hierarcgy_image.data_ptrs[layer_id] = (T *)malloc(hierarcgy_image.size[layer_id].width * hierarcgy_image.size[layer_id].height * sizeof(T));
		memset(hierarcgy_image.data_ptrs[layer_id], 0x00, hierarcgy_image.size[layer_id].width * hierarcgy_image.size[layer_id].height * sizeof(T));
	}
}
// Host memory release for hierarchy image
template <typename T>
void release_host_memory_for_hierarchy(Hierarchy_image<T> & hierarcgy_image)
{
	for (size_t layer_id = 0; layer_id < hierarcgy_image.number_of_layers; layer_id++)
		free(hierarcgy_image.data_ptrs[layer_id]);
}
#pragma endregion


// ---------------------------- Preprocess_engine
#pragma region(Preprocess_engine(base class))
void Preprocess_engine::init(My_Type::Vector2i raw_color_size, My_Type::Vector2i raw_depth_size,
							 int image_alignment_width,
							 int number_of_layers)
{
	// Set raw image size
	this->raw_color_size = raw_color_size;
	this->raw_depth_size = raw_depth_size;
	this->image_alignment_width = image_alignment_width;
	// Compute aligned depth image size
	this->aligned_depth_size.width = ceil_by_stride(this->raw_depth_size.width, this->image_alignment_width);
	this->aligned_depth_size.height = ceil_by_stride(this->raw_depth_size.height, this->image_alignment_width);
}



//
void Preprocess_engine::copy_previous_intensity_as_model()
{
	//
	for (size_t layer_id = 0; layer_id < this->dev_hierarchy_points.number_of_layers; layer_id++)
	{
		checkCudaErrors(cudaMemcpy(this->dev_hierarchy_model_intensity.data_ptrs[layer_id], this->dev_hierarchy_intensity.data_ptrs[layer_id],
			this->dev_hierarchy_intensity.size[layer_id].width * this->dev_hierarchy_intensity.size[layer_id].height * sizeof(float),
			cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(this->dev_hierarchy_model_gradient.data_ptrs[layer_id], this->dev_hierarchy_gradient.data_ptrs[layer_id],
			this->dev_hierarchy_gradient.size[layer_id].width * this->dev_hierarchy_gradient.size[layer_id].height * sizeof(My_Type::Vector2f),
			cudaMemcpyDeviceToDevice));

	}
}


#pragma endregion

// ---------------------------- Preprocess_RGBD
#pragma region(Preprocess_RGBD)

//
Preprocess_RGBD::~Preprocess_RGBD()
{
	// CUDA
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_intensity);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_gradient);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_intensity);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_gradient);
	checkCudaErrors(cudaFree(this->dev_raw_aligned_points));
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_points);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_normals);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_points);
	release_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_normals);
	checkCudaErrors(cudaFree(this->dev_raw_color));
	checkCudaErrors(cudaFree(this->dev_raw_depth));
	checkCudaErrors(cudaFree(this->dev_filtered_depth));

	// HOST
	free(this->raw_aligned_points);
	release_host_memory_for_hierarchy(this->hierarchy_points);
	release_host_memory_for_hierarchy(this->hierarchy_normals);
	release_host_memory_for_hierarchy(this->hierarchy_model_points);
	release_host_memory_for_hierarchy(this->hierarchy_model_normals);
	release_host_memory_for_hierarchy(this->hierarchy_intensity);
	release_host_memory_for_hierarchy(this->hierarchy_gradient);
	release_host_memory_for_hierarchy(this->hierarchy_model_intensity);
	release_host_memory_for_hierarchy(this->hierarchy_model_gradient);
}


//
void Preprocess_RGBD::init(My_Type::Vector2i raw_color_size, My_Type::Vector2i raw_depth_size,
						   int image_alignment_width,
						   int number_of_layers)
{
	// Init parameters
	Preprocess_engine::init(raw_color_size, raw_depth_size, image_alignment_width, number_of_layers);


	// Hierarchy images
	// Pre-compute hierarchy image memory occupancy
	this->dev_hierarchy_points.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_normals.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_model_points.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_model_normals.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_intensity.init_parameters(this->raw_color_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_gradient.init_parameters(this->raw_color_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_model_intensity.init_parameters(this->raw_color_size, this->image_alignment_width, number_of_layers);
	this->dev_hierarchy_model_gradient.init_parameters(this->raw_color_size, this->image_alignment_width, number_of_layers);
	// Allocate memory for hierarchy images
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_intensity);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_gradient);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_intensity);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_gradient);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_points);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_normals);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_points);
	allocate_CUDA_memory_for_hierarchy(this->dev_hierarchy_model_normals);


	// Allocate raw images CUDA buffer
	checkCudaErrors(cudaMalloc((void **)&(this->dev_raw_color),
		this->raw_color_size.width * this->raw_color_size.height * sizeof(RawColorType)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_raw_depth),
		this->raw_depth_size.width * this->raw_depth_size.height * sizeof(RawDepthType)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_depth_buffer),
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_filtered_depth),
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_raw_aligned_points),
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(My_Type::Vector3f)));
	// Memory initialization
	checkCudaErrors(cudaMemset(this->dev_raw_color, 0x00,
		this->raw_color_size.width * this->raw_color_size.height * sizeof(RawColorType)));
	checkCudaErrors(cudaMemset(this->dev_raw_depth, 0x00,
		this->raw_depth_size.width * this->raw_depth_size.height * sizeof(RawDepthType)));
	checkCudaErrors(cudaMemset(this->dev_depth_buffer, 0x00,
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(float)));
	checkCudaErrors(cudaMemset(this->dev_filtered_depth, 0x00,
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(float)));
	checkCudaErrors(cudaMemset(this->dev_raw_aligned_points, 0x00,
		this->aligned_depth_size.width * this->aligned_depth_size.height * sizeof(My_Type::Vector3f)));


	// HOST
	this->raw_aligned_points = (My_Type::Vector3f *)malloc(this->aligned_depth_size.width * this->aligned_depth_size.height
														   * sizeof(My_Type::Vector3f));
	this->hierarchy_intensity.init_parameters(this->raw_color_size, this->image_alignment_width, number_of_layers);
	this->hierarchy_points.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->hierarchy_normals.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->hierarchy_model_points.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	this->hierarchy_model_normals.init_parameters(this->raw_depth_size, this->image_alignment_width, number_of_layers);
	allocate_host_memory_for_hierarchy(this->hierarchy_intensity);
	allocate_host_memory_for_hierarchy(this->hierarchy_points);
	allocate_host_memory_for_hierarchy(this->hierarchy_normals);
	allocate_host_memory_for_hierarchy(this->hierarchy_model_points);
	allocate_host_memory_for_hierarchy(this->hierarchy_model_normals);
}


//
void Preprocess_RGBD::preprocess_image(cv::Mat & raw_color, cv::Mat & raw_depth)
{
	// Filter depth image TODO 双边滤波存在问题
	this->filter_image(raw_color, raw_depth);

	// Generate hierarchy image
	this->generate_hierarchy_image();

}


//
void Preprocess_RGBD::preprocess_model_points(My_Type::Vector3f * dev_model_points, My_Type::Vector3f * dev_model_normals)
{
	dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

	//
	checkCudaErrors(cudaMemcpy(this->dev_hierarchy_model_points.data_ptrs[0], dev_model_points,
		this->dev_hierarchy_model_points.size[0].width * this->dev_hierarchy_model_points.size[0].height * sizeof(My_Type::Vector3f),
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(this->dev_hierarchy_model_normals.data_ptrs[0], dev_model_normals,
		this->dev_hierarchy_model_normals.size[0].width * this->dev_hierarchy_model_normals.size[0].height * sizeof(My_Type::Vector3f),
		cudaMemcpyDeviceToDevice));


	// Generate hierarchy depth & normal image
	{
		for (size_t layer_id = 1; layer_id < this->dev_hierarchy_model_points.number_of_layers; layer_id++)
		{
			thread_rect.x = this->image_alignment_width;
			thread_rect.y = this->image_alignment_width;
			thread_rect.z = 1;
			block_rect.x = this->dev_hierarchy_points.size[layer_id].width / thread_rect.x;
			block_rect.y = this->dev_hierarchy_points.size[layer_id].height / thread_rect.y;
			block_rect.z = 1;
			// Lunch kernel function (points)
			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
											  this->dev_hierarchy_model_points.data_ptrs[layer_id - 1],
											  this->dev_hierarchy_model_points.size[layer_id - 1],
											  this->dev_hierarchy_model_points.data_ptrs[layer_id]);
//			CUDA_CKECK_KERNEL;
////			 Lunch kernel function (normals)
			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
											  this->dev_hierarchy_model_normals.data_ptrs[layer_id - 1],
											  this->dev_hierarchy_model_normals.size[layer_id - 1],
											  this->dev_hierarchy_model_normals.data_ptrs[layer_id]);
//			CUDA_CKECK_KERNEL;
		}
	}
}


//
void Preprocess_RGBD::filter_image(cv::Mat & raw_color, cv::Mat & raw_depth)
{
	dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

	// Copy to dev_raw_depth
	checkCudaErrors(cudaMemcpy(this->dev_raw_depth, raw_depth.data,
		this->raw_depth_size.width * this->raw_depth_size.height * sizeof(RawDepthType),
		cudaMemcpyHostToDevice));
	//
	checkCudaErrors(cudaMemcpy(this->dev_raw_color, raw_color.data,
		this->raw_color_size.width * this->raw_color_size.height * sizeof(RawColorType),
		cudaMemcpyHostToDevice));


	// ------------ Filter operation
	// Bilateral filter radius
	const int filter_radius = 5;
	if ((filter_radius & 0x00000001) == 0x00000001)
	{
		// Generate aligned depth image
        thread_rect.x = this->image_alignment_width;
        thread_rect.y = this->image_alignment_width;
        thread_rect.z = 1;
        block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
        block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
        block_rect.z = 1;
        // Lunch kernel function
        generate_float_type_depth_CUDA(block_rect, thread_rect,
                                       this->dev_raw_depth,
                                       SLAM_system_settings::instance()->sensor_params,
                                       this->raw_depth_size,
                                       this->dev_depth_buffer);
        //CUDA_CKECK_KERNEL;


    // Bilateral filter

        thread_rect.x = this->image_alignment_width;
        thread_rect.y = this->image_alignment_width;
        thread_rect.z = 1;
        block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
        block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
        block_rect.z = 1;
        // Lunch kernel function
        bilateral_filter_5x5_CUDA(block_rect, thread_rect,
                                  this->dev_depth_buffer,
                                  this->dev_filtered_depth);
        //CUDA_CKECK_KERNEL;

	}
	else
	{
		// Generate aligned depth image
		{
			thread_rect.x = this->image_alignment_width;
			thread_rect.y = this->image_alignment_width;
			thread_rect.z = 1;
			block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
			block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
			block_rect.z = 1;
			// Lunch kernel function
			generate_float_type_depth_CUDA(block_rect, thread_rect,
										   this->dev_raw_depth,
										   SLAM_system_settings::instance()->sensor_params,
										   this->raw_depth_size,
										   this->dev_filtered_depth);
			//CUDA_CKECK_KERNEL;
		}
	}

	// Perform cascade bilateral filter process
	int repeat_times = filter_radius >> 1;
	for (int i = 0; i < repeat_times; i++)
	{
		thread_rect.x = this->image_alignment_width;
		thread_rect.y = this->image_alignment_width;
		thread_rect.z = 1;
		block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
		block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
		block_rect.z = 1;
		// Lunch kernel function
		bilateral_filter_5x5_CUDA(block_rect, thread_rect,
								  this->dev_filtered_depth,
								  this->dev_depth_buffer);
		//CUDA_CKECK_KERNEL;
		bilateral_filter_5x5_CUDA(block_rect, thread_rect,
								  this->dev_depth_buffer,
								  this->dev_filtered_depth);
		//CUDA_CKECK_KERNEL;
	}


	// Generate aligned intensity image for camera pose tracking

    thread_rect.x = this->image_alignment_width;
    thread_rect.y = this->image_alignment_width;
    thread_rect.z = 1;
    block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
    block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
    block_rect.z = 1;
    // Lunch kernel function
    generate_intensity_image_CUDA(block_rect, thread_rect,
                                  this->dev_raw_color,
                                  this->raw_color_size,
                                  this->dev_hierarchy_intensity.data_ptrs[0]);
    //CUDA_CKECK_KERNEL;

	// Filter intensity image
	{

	}

	// Compute gradient for intensity image
	{
		thread_rect.x = this->image_alignment_width;
		thread_rect.y = this->image_alignment_width;
		thread_rect.z = 1;
		block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
		block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
		block_rect.z = 1;
		// Lunch kernel function
		generate_gradient_image_CUDA(block_rect, thread_rect,
									 this->dev_hierarchy_intensity.data_ptrs[0],
									 this->dev_hierarchy_gradient.data_ptrs[0]);
		//CUDA_CKECK_KERNEL;
	}

}


//
void Preprocess_RGBD::generate_hierarchy_image()
{
	dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

	// ----- Layer 0 -----
	// Generate aligned points image (From filtered depth image) for camera pose tracking
	{
		thread_rect.x = this->image_alignment_width;
		thread_rect.y = this->image_alignment_width;
		thread_rect.z = 1;
		block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
		block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
		block_rect.z = 1;
		// Lunch kernel function
		generate_aligned_points_image_CUDA(block_rect, thread_rect,
										   this->dev_filtered_depth,
										   SLAM_system_settings::instance()->sensor_params,
										   this->dev_hierarchy_points.data_ptrs[0]);
		//CUDA_CKECK_KERNEL;
	}
	// Generate aligned points image (From raw depth image) for map update
	{
		thread_rect.x = this->image_alignment_width;
		thread_rect.y = this->image_alignment_width;
		thread_rect.z = 1;
		block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
		block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
		block_rect.z = 1;
		// Lunch kernel function
		generate_aligned_points_image_CUDA(block_rect, thread_rect,
										   this->dev_filtered_depth,
										   SLAM_system_settings::instance()->sensor_params,
										   this->dev_raw_aligned_points);
		//CUDA_CKECK_KERNEL;
	}
	// Generate aligned normal image (From hierarchy points)
	// -------------------------------- To Do : normal consecutive validation
	{
		thread_rect.x = this->image_alignment_width;
		thread_rect.y = this->image_alignment_width;
		thread_rect.z = 1;
		block_rect.x = this->dev_hierarchy_points.size[0].width / thread_rect.x;
		block_rect.y = this->dev_hierarchy_points.size[0].height / thread_rect.y;
		block_rect.z = 1;
		// Lunch kernel function
		compute_normals_image_CUDA(block_rect, thread_rect,
								   this->dev_hierarchy_points.data_ptrs[0],
								   this->dev_hierarchy_normals.data_ptrs[0]);
		//CUDA_CKECK_KERNEL;
	}



	// ----- Layer 1 to n -----
	// Generate hierarchy intensity image
//	{
//		for (size_t layer_id = 1; layer_id < this->dev_hierarchy_points.number_of_layers; layer_id++)
//		{
//			thread_rect.x = this->image_alignment_width;
//			thread_rect.y = this->image_alignment_width;
//			thread_rect.z = 1;
//			block_rect.x = this->dev_hierarchy_intensity.size[layer_id].width / thread_rect.x;
//			block_rect.y = this->dev_hierarchy_intensity.size[layer_id].height / thread_rect.y;
//			block_rect.z = 1;
//            //TODO 存在bug
//			// Lunch kernel function (points)
//			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
//											  this->dev_hierarchy_intensity.data_ptrs[layer_id - 1],
//											  this->dev_hierarchy_intensity.size[layer_id - 1],
//											  this->dev_hierarchy_intensity.data_ptrs[layer_id]);
//			//CUDA_CKECK_KERNEL;
//			// Lunch kernel function (normals)
//			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
//											  this->dev_hierarchy_gradient.data_ptrs[layer_id - 1],
//											  this->dev_hierarchy_gradient.size[layer_id - 1],
//											  this->dev_hierarchy_gradient.data_ptrs[layer_id]);
//			//CUDA_CKECK_KERNEL;
//		}
//	}
	// Generate hierarchy depth & normal image
	{
		for (size_t layer_id = 1; layer_id < this->dev_hierarchy_points.number_of_layers; layer_id++)
		{
			thread_rect.x = this->image_alignment_width;
			thread_rect.y = this->image_alignment_width;
			thread_rect.z = 1;
			block_rect.x = this->dev_hierarchy_points.size[layer_id].width / thread_rect.x;
			block_rect.y = this->dev_hierarchy_points.size[layer_id].height / thread_rect.y;
			block_rect.z = 1;
			//TODO 存在bug
			// Lunch kernel function (points)
			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
											  this->dev_hierarchy_points.data_ptrs[layer_id - 1],
											  this->dev_hierarchy_points.size[layer_id - 1],
											  this->dev_hierarchy_points.data_ptrs[layer_id]);
			//CUDA_CKECK_KERNEL;
			// Lunch kernel function (normals)
			down_sample_hierarchy_layers_CUDA(block_rect, thread_rect,
											  this->dev_hierarchy_normals.data_ptrs[layer_id - 1],
											  this->dev_hierarchy_normals.size[layer_id - 1],
											  this->dev_hierarchy_normals.data_ptrs[layer_id]);
			//CUDA_CKECK_KERNEL;
		}
	}


	// For feature detection
	if (true)
	{
		//
		for (size_t layer_id = 0; layer_id < this->dev_hierarchy_points.number_of_layers; layer_id++)
		{
			checkCudaErrors(cudaMemcpy(this->hierarchy_points.data_ptrs[layer_id], this->dev_hierarchy_points.data_ptrs[layer_id],
				this->dev_hierarchy_points.size[layer_id].width * this->dev_hierarchy_points.size[layer_id].height * sizeof(My_Type::Vector3f),
				cudaMemcpyDeviceToHost));
		}
	}
}




//
void Preprocess_RGBD::generate_render_information()
{
	//
	for (size_t layer_id = 0; layer_id < this->dev_hierarchy_points.number_of_layers; layer_id++)
	{
		checkCudaErrors(cudaMemcpy(this->hierarchy_points.data_ptrs[layer_id], this->dev_hierarchy_points.data_ptrs[layer_id],
			this->dev_hierarchy_points.size[layer_id].width * this->dev_hierarchy_points.size[layer_id].height * sizeof(My_Type::Vector3f),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hierarchy_normals.data_ptrs[layer_id], this->dev_hierarchy_normals.data_ptrs[layer_id],
			this->dev_hierarchy_normals.size[layer_id].width * this->dev_hierarchy_normals.size[layer_id].height * sizeof(My_Type::Vector3f),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hierarchy_model_points.data_ptrs[layer_id], this->dev_hierarchy_model_points.data_ptrs[layer_id],
			this->dev_hierarchy_model_points.size[layer_id].width * this->dev_hierarchy_model_points.size[layer_id].height * sizeof(My_Type::Vector3f),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hierarchy_model_normals.data_ptrs[layer_id], this->dev_hierarchy_model_normals.data_ptrs[layer_id],
			this->dev_hierarchy_model_normals.size[layer_id].width * this->dev_hierarchy_model_normals.size[layer_id].height * sizeof(My_Type::Vector3f),
			cudaMemcpyDeviceToHost));
	}
}

#pragma endregion


