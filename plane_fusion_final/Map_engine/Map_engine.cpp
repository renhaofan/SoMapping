#include "Map_engine.h"


// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "OurLib/my_math_functions.h"
#include "SLAM_system/SLAM_system_settings.h"
#include "UI_engine/UI_parameters.h"


// ------------------------- Map_engine 
#pragma region(Map engine base)
Map_engine::~Map_engine()
{
	//
	checkCudaErrors(cudaFree(this->dev_model_points));
	checkCudaErrors(cudaFree(this->dev_model_normals));
	checkCudaErrors(cudaFree(this->dev_model_weight));
	checkCudaErrors(cudaFree(this->dev_model_plane_labels));
	//
	free(this->scene_points);
	free(this->scene_normals);
	free(this->scene_weight);
	free(this->scene_plane_labels);
}


//
void Map_engine::init_base()
{
	//
	int ceil_depth_image_width = ceil_by_stride(SLAM_system_settings::instance()->aligned_depth_size.width,
												SLAM_system_settings::instance()->image_alginment_patch_width);
	int ceil_depth_image_height = ceil_by_stride(SLAM_system_settings::instance()->aligned_depth_size.height,
												 SLAM_system_settings::instance()->image_alginment_patch_width);
	int ceil_viewport_width = ceil_by_stride(UI_parameters::instance()->main_viewport_size.width,
											 SLAM_system_settings::instance()->image_alginment_patch_width);
	int ceil_viewport_height = ceil_by_stride(UI_parameters::instance()->main_viewport_size.height,
											  SLAM_system_settings::instance()->image_alginment_patch_width);
	
	
	// 
	checkCudaErrors(cudaMalloc((void **)&(this->dev_model_points),
		ceil_depth_image_width * ceil_depth_image_height * sizeof(My_Type::Vector3f)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_model_normals),
		ceil_depth_image_width * ceil_depth_image_height * sizeof(My_Type::Vector3f)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_model_weight),
		ceil_depth_image_width * ceil_depth_image_height * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&(this->dev_model_plane_labels),
		ceil_depth_image_width * ceil_depth_image_height * sizeof(int)));
	//
	this->scene_points = (My_Type::Vector3f *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(My_Type::Vector3f));
	this->scene_normals = (My_Type::Vector3f *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(My_Type::Vector3f));
	this->scene_weight = (int *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(int));
	this->scene_plane_labels = (int *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(int));



}


//
void Map_engine::reshape_render_viewport(My_Type::Vector2i viewport_size)
{
	// Release memory
	if (this->scene_points)			free(this->scene_points);
	if (this->scene_normals)		free(this->scene_normals);
	if (this->scene_weight)			free(this->scene_weight);
	if (this->scene_plane_labels)	free(this->scene_plane_labels);

	// Re-allocate memory
	int ceil_viewport_width = ceil_by_stride(viewport_size.width,
											 SLAM_system_settings::instance()->image_alginment_patch_width);
	int ceil_viewport_height = ceil_by_stride(viewport_size.height,
											  SLAM_system_settings::instance()->image_alginment_patch_width);
	//
	this->scene_points = (My_Type::Vector3f *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(My_Type::Vector3f));
	this->scene_normals = (My_Type::Vector3f *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(My_Type::Vector3f));
	this->scene_weight = (int *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(int));
	this->scene_plane_labels = (int *)malloc(ceil_viewport_width * ceil_viewport_height * sizeof(int));
	
}
#pragma endregion


// ------------------------- Basic_Voxel_map 
#pragma region(Basic voxel map)
// 
Basic_Voxel_map::Basic_Voxel_map()
{
	this->voxel_map_ptr = new Voxel_map();
	this->plane_map_ptr = new Plane_map();
}
//
Basic_Voxel_map::~Basic_Voxel_map()
{
	delete this->voxel_map_ptr;
	delete this->plane_map_ptr;
}


//
void Basic_Voxel_map::init_map()
{
	this->init_base();

	this->voxel_map_ptr->init_Voxel_map(SLAM_system_settings::instance()->aligned_depth_size,
										SUBMAP_VOXEL_BLOCK_NUM);

	this->plane_map_ptr->init();
}




//
void Basic_Voxel_map::update_map_after_tracking(My_pose & camera_pose,
												My_Type::Vector3f * dev_current_points,
												My_Type::Vector3f * dev_current_normals,
												int * dev_plane_labels)
{
	this->voxel_map_ptr->allocate_voxel_block(dev_current_points, camera_pose.mat);
	
	this->voxel_map_ptr->fusion_SDF_to_voxel(dev_current_points, dev_current_normals, camera_pose.mat);
	
	if (!this->voxel_map_fusion_initialize_done)
	{
		this->voxel_map_fusion_initialize_done = true;
		for (int repeat_fusion_i = 0; repeat_fusion_i < MIN_RAYCAST_WEIGHT; repeat_fusion_i++)
			this->voxel_map_ptr->fusion_SDF_to_voxel(dev_current_points, dev_current_normals, camera_pose.mat);
	}

	// Update plane label
	if (dev_plane_labels != nullptr)
	{
		this->voxel_map_ptr->fusion_plane_label_to_voxel(dev_current_points, dev_plane_labels, camera_pose.mat);
	}
}


//
void Basic_Voxel_map::update_plane_map(const Plane_info * current_planes,
									   std::vector<My_Type::Vector2i> & matches)
{
	this->plane_map_ptr->update_plane_list(current_planes, matches);
}


//
void Basic_Voxel_map::generate_plane_map()
{
	
	this->plane_map_ptr->generate_plane_map(this->voxel_map_ptr->dev_entrise, 
											this->voxel_map_ptr->dev_voxel_block_array);
}


//
void Basic_Voxel_map::raycast_points_from_map(Eigen::Matrix4f & camera_pose,
											  RaycastMode raycast_mode)
{
	//
	this->voxel_map_ptr->raycast_by_pose(camera_pose, raycast_mode);
	// 
	if (raycast_mode == RaycastMode::RAYCAST_FOR_VIEW)
	{
		checkCudaErrors(cudaMemcpy(this->scene_points, this->voxel_map_ptr->dev_scene_points,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_normals, this->voxel_map_ptr->dev_scene_normals,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_weight, this->voxel_map_ptr->dev_scene_weight,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_plane_labels, this->voxel_map_ptr->dev_scene_plane_label,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(int), cudaMemcpyDeviceToHost));

	}
	else
	{
		checkCudaErrors(cudaMemcpy(this->dev_model_points, this->voxel_map_ptr->dev_raycast_points,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(this->dev_model_normals, this->voxel_map_ptr->dev_raycast_normal,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(this->dev_model_plane_labels, this->voxel_map_ptr->dev_raycast_plane_label,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(int), cudaMemcpyDeviceToDevice));
	}
}
#pragma endregion


// ------------------------- Submap_Voxel_map
#pragma region(Submap voxel map)

// 
Submap_Voxel_map::Submap_Voxel_map()
{
	this->voxel_map_ptr = new Voxel_map();
	this->plane_map_ptr = new Plane_map();
}
//
Submap_Voxel_map::~Submap_Voxel_map()
{
	delete this->voxel_map_ptr;
	delete this->plane_map_ptr;
}

//
void Submap_Voxel_map::init_map()
{
	//
	this->init_base();

	// Initiate Voxel_map
	this->voxel_map_ptr->init_Voxel_map(SLAM_system_settings::instance()->aligned_depth_size,
										SUBMAP_VOXEL_BLOCK_NUM);
	//
	this->plane_map_ptr->init();
}


//
void Submap_Voxel_map::update_map_form_last_map(My_pose & camera_pose,
												My_Type::Vector3f * dev_current_points,
												HashEntry * dev_entries,
												Voxel_f * dev_voxel_array)
{
	//
	this->voxel_map_ptr->update_from_last_voxel_map(camera_pose, dev_current_points, dev_entries, dev_voxel_array);
}


//
void Submap_Voxel_map::update_map_after_tracking(My_pose & camera_pose,
												 My_Type::Vector3f * dev_current_points,
												 My_Type::Vector3f * dev_current_normals,
												 int * dev_plane_labels)
{

	// -------------- TODO add submap 
	this->frame_counter++;
	// Allocation
	int number_of_blocks = this->voxel_map_ptr->allocate_voxel_block(dev_current_points, camera_pose.mat);
	// Update
	this->voxel_map_ptr->fusion_SDF_to_voxel(dev_current_points, dev_current_normals, camera_pose.mat);


	//
	if (first_frame_to_this_submap)
	{
		first_frame_to_this_submap = false;
		this->init_number_of_blocks = number_of_blocks;
		//printf("this->init_number_of_blocks = %d\n", this->init_number_of_blocks);
		//
		for (int repeat_fusion_i = 0; repeat_fusion_i <= MIN_RAYCAST_WEIGHT; repeat_fusion_i++)
			this->voxel_map_ptr->fusion_SDF_to_voxel(dev_current_points, dev_current_normals, camera_pose.mat);
	}

	// Update plane label
	if (dev_plane_labels != nullptr)
	{
		this->voxel_map_ptr->fusion_plane_label_to_voxel(dev_current_points, dev_plane_labels, camera_pose.mat);
	}
}


//
void Submap_Voxel_map::update_plane_map(const Plane_info * current_planes,
										std::vector<My_Type::Vector2i> & matches)
{
	this->plane_map_ptr->update_plane_list(current_planes, matches);
}



//
void Submap_Voxel_map::raycast_points_from_map(Eigen::Matrix4f & camera_pose,
											   RaycastMode raycast_mode)
{
	//
	this->voxel_map_ptr->raycast_by_pose(camera_pose, raycast_mode);
	// 
	if (raycast_mode == RaycastMode::RAYCAST_FOR_VIEW)
	{
		checkCudaErrors(cudaMemcpy(this->scene_points, this->voxel_map_ptr->dev_scene_points,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_normals, this->voxel_map_ptr->dev_scene_normals,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_weight, this->voxel_map_ptr->dev_scene_weight,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->scene_plane_labels, this->voxel_map_ptr->dev_scene_plane_label,
			this->voxel_map_ptr->scene_depth_width * this->voxel_map_ptr->scene_depth_height * sizeof(int), cudaMemcpyDeviceToHost));

	}
	else
	{
		checkCudaErrors(cudaMemcpy(this->dev_model_points, this->voxel_map_ptr->dev_raycast_points,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(this->dev_model_normals, this->voxel_map_ptr->dev_raycast_normal,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(My_Type::Vector3f), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(this->dev_model_plane_labels, this->voxel_map_ptr->dev_raycast_plane_label,
			this->voxel_map_ptr->raycast_depth_width * this->voxel_map_ptr->raycast_depth_height * sizeof(int), cudaMemcpyDeviceToDevice));
	}
}



//
bool Submap_Voxel_map::consider_to_create_new_submap()
{
	//
	if (this->voxel_map_ptr->number_of_blocks > this->init_number_of_blocks * 2.0 &&
		this->frame_counter > 200)
	{	return true;	}
	else
	{	return false;	}
}


//
void Submap_Voxel_map::compress_voxel_map()
{
	// Compress voxel map (re-allocate Voxel-Block-Array)
	this->voxel_map_ptr->compress_voxel_map();

}


//
void Submap_Voxel_map::release_voxel_map()
{
	// Compress voxel map (re-allocate Voxel-Block-Array)
	this->voxel_map_ptr->release_voxel_map();

}

//
void Submap_Voxel_map::generate_plane_map()
{

	this->plane_map_ptr->generate_plane_map(this->voxel_map_ptr->dev_entrise,
											this->voxel_map_ptr->dev_voxel_block_array);
}


#pragma endregion

