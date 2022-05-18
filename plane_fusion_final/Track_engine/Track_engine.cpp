

#include "Track_engine.h"



// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "Solver_functor.h"
// CUDA functions
#include "Track_KernelFunc.cuh"


#include <iostream>
using namespace std;


//
void copy_out_hessian_from_accumulate_data(Eigen::Matrix<float, 6, 6> & out_hessian,
										  Eigen::Matrix<float, 6, 1> & out_nabla,
										  const Accumulate_result & accu_data)
{
	float * ptr_f = nullptr;

	// Hessian
	int counter = 0;
	ptr_f = (float *)out_hessian.data();
	for (int i = 0; i < 6; i++)	for (int j = 0; j <= i; j++, counter++)
	{
		out_hessian(i, j) = accu_data.hessian_upper[counter];
		out_hessian(j, i) = accu_data.hessian_upper[counter];
	}
	// Nabla
	ptr_f = (float *)out_nabla.data();
	for (int i = 0; i < 6; i++)
	{
		ptr_f[i] = accu_data.nabla[i];
	}
}



// ------------------------- Track_engine (base class / virtual interface)
#pragma region(Track_engine)

Track_engine::Track_engine()
{
}
Track_engine::~Track_engine()
{
#ifdef COMPILE_DEBUG_CODE
	free(this->correspondence_lines);
	checkCudaErrors(cudaFree(this->dev_correspondence_lines));
#endif
}

void Track_engine::init()
{
#ifdef COMPILE_DEBUG_CODE
	int buffer_size = SLAM_system_settings::instance()->aligned_depth_size.width * SLAM_system_settings::instance()->aligned_depth_size.height * sizeof(My_Type::Vector3f) * 2;
	this->correspondence_lines = (My_Type::Vector3f *)malloc(buffer_size);
	checkCudaErrors(cudaMalloc((void **)&(this->dev_correspondence_lines), buffer_size));
#endif
}




void Track_engine::update_camera_pose(Eigen::Matrix4f & camera_pose) const 
{
	switch (this->tracking_state)
	{
		case TrackingState::TRACKINE_SUCCED:
		{
			camera_pose = camera_pose.eval() * this->incremental_pose;
			break;
		}
		case TrackingState::BEFORE_TRACKING:
		case TrackingState::DURING_TRACKING:
		case TrackingState::TRACKING_FAILED:
		default:
			break;
	}
}


#ifdef COMPILE_DEBUG_CODE
void Track_engine::generate_icp_correspondence_lines(const Hierarchy_image<My_Type::Vector3f> & dev_current_points_hierarchy,
													 const Hierarchy_image<My_Type::Vector3f> & dev_model_points_hierarchy,
													 const Hierarchy_image<My_Type::Vector3f> & dev_current_normals_hierarchy,
													 const Hierarchy_image<My_Type::Vector3f> & dev_model_normals_hierarchy)
{
	My_Type::Matrix44f cache_increment_pose;
	memcpy(cache_increment_pose.data, this->incremental_pose.data(), 4 * 4 * sizeof(float));

	//
	dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);
	thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
	thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
	thread_rect.z = 1;
	block_rect.x = dev_current_points_hierarchy.size[0].width / thread_rect.x;
	block_rect.y = dev_current_points_hierarchy.size[0].height / thread_rect.y;
	block_rect.z = 1;
	generate_correspondence_lines_CUDA(block_rect, thread_rect,
									   dev_current_points_hierarchy.data_ptrs[0],
									   dev_model_points_hierarchy.data_ptrs[0],
									   dev_current_normals_hierarchy.data_ptrs[0],
									   dev_model_normals_hierarchy.data_ptrs[0],
									   SLAM_system_settings::instance()->sensor_params,
									   cache_increment_pose,
									   this->dev_correspondence_lines);
	//CUDA_CKECK_KERNEL;


	// Copy out result 
	int buffer_size = SLAM_system_settings::instance()->aligned_depth_size.width * SLAM_system_settings::instance()->aligned_depth_size.height * sizeof(My_Type::Vector3f) * 2;
	checkCudaErrors(cudaMemcpy(this->correspondence_lines, this->dev_correspondence_lines, buffer_size,	cudaMemcpyDeviceToHost));
}
#endif

#pragma endregion


// ------------------------- Basic_ICP_tracker
#pragma region(Basic_ICP_Tracker)
Basic_ICP_tracker::Basic_ICP_tracker()
{
	checkCudaErrors(cudaMalloc((void **)&(this->dev_accumulation_buffer), sizeof(Accumulate_result)));
}
Basic_ICP_tracker::~Basic_ICP_tracker()
{
	checkCudaErrors(cudaFree(this->dev_accumulation_buffer));
}


//
TrackingState Basic_ICP_tracker::track_camera_pose(const Hierarchy_image<My_Type::Vector3f> & dev_current_points_hierarchy,
												   const Hierarchy_image<My_Type::Vector3f> & dev_model_points_hierarchy, 
												   const Hierarchy_image<My_Type::Vector3f> & dev_current_normals_hierarchy,
												   const Hierarchy_image<My_Type::Vector3f> & dev_model_normals_hierarchy,
												   const Hierarchy_image<float> & dev_current_intensity_hierarchy,
												   const Hierarchy_image<float> & dev_model_intensity_hierarchy,
												   const Hierarchy_image<My_Type::Vector2f> & dev_model_gradient_hierarchy,
												   Eigen::Matrix4f & camera_pose)
{
	// iteration times of each layer
	int iterate_times[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	// Functor of nonlinear solver
	//GaussianNewton_solver my_solver;
	LevenbergMarquardt_solver my_solver;

	//
	this->prepare_to_track_new_frame();

	//
	//for (int layer_id = 0;
	for (int layer_id = SLAM_system_settings::instance()->number_of_hierarchy_layers - 1;
		 layer_id >= 0; layer_id--)
	{
		//
		bool is_convergenced = false;
		//
		dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

		//
		for (iterate_times[layer_id] = 0;
			 iterate_times[layer_id] < SLAM_system_settings::instance()->max_iterate_times[layer_id]; 
			 iterate_times[layer_id]++)
		{
			//
			Eigen::Matrix<float, 6, 6> icp_hessian, photometric_hessian;
			Eigen::Matrix<float, 6, 1> icp_nabla, photometric_nabla;

			//
			this->prepare_to_new_iteration();

			// ----- Residual computation (Hierarchy_image --> Hessian, Nabla)
			{
				// Point-to-Plane ICP residual
				thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.z = 1;
				block_rect.x = dev_current_points_hierarchy.size[layer_id].width / thread_rect.x;
				block_rect.y = dev_current_points_hierarchy.size[layer_id].height / thread_rect.y;
				block_rect.z = 1;
				// Lunch kernel function
				compute_points_residual_CUDA(block_rect, thread_rect,
											 dev_current_points_hierarchy.data_ptrs[layer_id],
											 dev_model_points_hierarchy.data_ptrs[layer_id],
											 dev_current_normals_hierarchy.data_ptrs[layer_id],
											 dev_model_normals_hierarchy.data_ptrs[layer_id],
											 NULL,
											 SLAM_system_settings::instance()->sensor_params,
											 this->cache_increment_pose,
											 layer_id,
											 this->dev_accumulation_buffer);
				//CUDA_CKECK_KERNEL;

				// Copy out result 
				checkCudaErrors(cudaMemcpy((void *)&this->icp_accumulation, this->dev_accumulation_buffer,	sizeof(Accumulate_result),
					cudaMemcpyDeviceToHost));
				copy_out_hessian_from_accumulate_data(icp_hessian, icp_nabla, this->icp_accumulation);
				// Normalize residual
				float hessian_nrom = icp_hessian.norm();
				icp_hessian /= hessian_nrom;
				icp_nabla /= hessian_nrom;
			}


			// ----- Residual computation (Hierarchy_image --> Hessian, Nabla)
			{
				// Point-to-Plane ICP residual
				thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.z = 1;
				block_rect.x = dev_current_points_hierarchy.size[layer_id].width / thread_rect.x;
				block_rect.y = dev_current_points_hierarchy.size[layer_id].height / thread_rect.y;
				block_rect.z = 1;
				// Lunch kernel function
				compute_photometric_residual_CUDA(block_rect, thread_rect,
												  dev_current_points_hierarchy.data_ptrs[layer_id],
												  dev_current_intensity_hierarchy.data_ptrs[layer_id],
												  dev_model_intensity_hierarchy.data_ptrs[layer_id],
												  dev_model_gradient_hierarchy.data_ptrs[layer_id],
												  NULL,
												  SLAM_system_settings::instance()->sensor_params,
												  this->cache_increment_pose,
												  layer_id,
												  this->dev_accumulation_buffer);
				//CUDA_CKECK_KERNEL;

				// Copy out result 
				checkCudaErrors(cudaMemcpy((void *)&this->icp_accumulation, this->dev_accumulation_buffer, sizeof(Accumulate_result),
					cudaMemcpyDeviceToHost));
				copy_out_hessian_from_accumulate_data(photometric_hessian, photometric_nabla, this->icp_accumulation);
				// Normalize residual
				float hessian_nrom = photometric_hessian.norm();
				photometric_hessian /= hessian_nrom;
				photometric_nabla /= hessian_nrom;
			}


			// ----- Validate number of point pairs
			if (this->icp_accumulation.number_of_pairs <= 10)
			{
				return TrackingState::TRACKING_FAILED;
			}
			// ----- Solver (Hessian, Nabla --> X=[rx,ry,rz,tx,ty,tz] --> increment pose matrix)
			//
			{
				float photometric_weight = 0.5;
				// 
				this->total_hessian = icp_hessian + photometric_weight * photometric_weight * photometric_hessian;
				this->total_nabla = -(icp_nabla + photometric_weight * photometric_nabla);

				// Compute average point distance
				float average_distance = this->icp_accumulation.energy / (float)this->icp_accumulation.number_of_pairs;
				

				//
				//IterationState state = my_solver(this->total_hessian, this->total_nabla, this->incremental_pose);
				IterationState state = my_solver(this->total_hessian, this->total_nabla, 
												 average_distance, this->icp_accumulation.number_of_pairs,
												 this->incremental_pose);
				if (state == IterationState::FAILED || state == IterationState::CONVERGENCED)
				{
					//printf("%f\n", average_energy);
					break;
				}
			}

		}

	}
	// Update camera pose
	camera_pose = camera_pose.eval() * this->incremental_pose;

	return TrackingState::TRACKINE_SUCCED;
}


//
void Basic_ICP_tracker::prepare_to_track_new_frame()
{
	// Set tracking state
	this->tracking_state = TrackingState::BEFORE_TRACKING;
	this->incremental_pose.setIdentity();
}

//
void Basic_ICP_tracker::prepare_to_new_iteration()
{
	//
	this->total_hessian.setZero();
	this->total_nabla.setZero();

	// 
	checkCudaErrors(cudaMemset(this->dev_accumulation_buffer, 0x00, sizeof(Accumulate_result)));

	//
	memcpy(cache_increment_pose.data, this->incremental_pose.data(), 4 * 4 * sizeof(float));
}

#pragma endregion



// Keypoint + photometric + ICP
#pragma region(Keypoint Tracker)

Keypoint_ICP_tracker::Keypoint_ICP_tracker()
{
	checkCudaErrors(cudaMalloc((void **)&(this->dev_accumulation_buffer), sizeof(Accumulate_result)));
}
Keypoint_ICP_tracker::~Keypoint_ICP_tracker()
{
	checkCudaErrors(cudaFree(this->dev_accumulation_buffer));
}


//
TrackingState Keypoint_ICP_tracker::track_camera_pose(const Hierarchy_image<My_Type::Vector3f> & dev_current_points_hierarchy,
													  const Hierarchy_image<My_Type::Vector3f> & dev_model_points_hierarchy,
													  const Hierarchy_image<My_Type::Vector3f> & dev_current_normals_hierarchy,
													  const Hierarchy_image<My_Type::Vector3f> & dev_model_normals_hierarchy,
													  const Hierarchy_image<float> & dev_current_intensity_hierarchy,
													  const Hierarchy_image<float> & dev_model_intensity_hierarchy,
													  const Hierarchy_image<My_Type::Vector2f> & dev_model_gradient_hierarchy,
													  const std::vector<Eigen::Vector3f> & current_keypoints,
													  const std::vector<Eigen::Vector3f> & model_keypoints,
													  Eigen::Matrix4f & camera_pose)
{
	bool debug_flag = true;
	// iteration times of each layer
	int iterate_times[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	// Functor of nonlinear solver
	//GaussianNewton_solver my_solver;
	LevenbergMarquardt_solver my_solver;

	//
	this->prepare_to_track_new_frame();

	//
	IterationState state;
	//
	//for (int layer_id = 0;
	for (int layer_id = SLAM_system_settings::instance()->number_of_hierarchy_layers - 1;
		 layer_id >= 0; layer_id--)
	{
		//
		bool is_convergenced = false;
		//
		dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

		//
		for (iterate_times[layer_id] = 0;
			 iterate_times[layer_id] < SLAM_system_settings::instance()->max_iterate_times[layer_id];
			 iterate_times[layer_id]++)
		{
			//
			Eigen::Matrix<float, 6, 6> icp_hessian, photometric_hessian, keypoint_hessian;
			Eigen::Matrix<float, 6, 1> icp_nabla, photometric_nabla, keypoint_nabla;

			this->prepare_to_new_iteration();

			//
			float icp_weight = 1.0;
			float photometric_weight = 0;
			float keypoint_weight = 0.5;
			// ----- Residual computation (Hierarchy_image --> Hessian, Nabla)
			{
				// Point-to-Plane ICP residual
				thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.z = 1;
				block_rect.x = dev_current_points_hierarchy.size[layer_id].width / thread_rect.x;
				block_rect.y = dev_current_points_hierarchy.size[layer_id].height / thread_rect.y;
				block_rect.z = 1;
				// Lunch kernel function
				compute_points_residual_CUDA(block_rect, thread_rect,
											 dev_current_points_hierarchy.data_ptrs[layer_id],
											 dev_model_points_hierarchy.data_ptrs[layer_id],
											 dev_current_normals_hierarchy.data_ptrs[layer_id],
											 dev_model_normals_hierarchy.data_ptrs[layer_id],
											 NULL,
											 SLAM_system_settings::instance()->sensor_params,
											 this->cache_increment_pose,
											 layer_id,
											 this->dev_accumulation_buffer);
				//CUDA_CKECK_KERNEL;

				// Copy out result 
				checkCudaErrors(cudaMemcpy((void *)&this->icp_accumulation, this->dev_accumulation_buffer, sizeof(Accumulate_result),
					cudaMemcpyDeviceToHost));
				copy_out_hessian_from_accumulate_data(icp_hessian, icp_nabla, this->icp_accumulation);
				// Normalize residual
				if (this->icp_accumulation.number_of_pairs > 100)
				{
					icp_hessian /= this->icp_accumulation.number_of_pairs;
					icp_nabla /= this->icp_accumulation.number_of_pairs;
				}
				else
				{
					icp_hessian.setIdentity();
					icp_nabla.setZero();
				}

				// Check icp condition number
				float hessian_norm_infinit = icp_hessian.lpNorm<Eigen::Infinity>();
				float hessian_norm_infinit_inv = icp_hessian.inverse().lpNorm<Eigen::Infinity>();
				float conditional_num = hessian_norm_infinit * hessian_norm_infinit_inv;
				if (debug_flag)
				{
					debug_flag = false;
					//printf("conditional_num = %f\n", conditional_num);
				}
				if (conditional_num < 1000 && true)
				{
					// Geometry is confidence enough
					icp_weight = 1.0;
					photometric_weight = 0.0;
					keypoint_weight = 0.0;
					goto Solve_one_iteration;
				}
			}

			// ----- Residual computation (Hierarchy_image --> Hessian, Nabla)
			{
				// Point-to-Plane ICP residual
				thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
				thread_rect.z = 1;
				block_rect.x = dev_current_points_hierarchy.size[layer_id].width / thread_rect.x;
				block_rect.y = dev_current_points_hierarchy.size[layer_id].height / thread_rect.y;
				block_rect.z = 1;
				// Lunch kernel function
				compute_photometric_residual_CUDA(block_rect, thread_rect,
												  dev_current_points_hierarchy.data_ptrs[layer_id],
												  dev_current_intensity_hierarchy.data_ptrs[layer_id],
												  dev_model_intensity_hierarchy.data_ptrs[layer_id],
												  dev_model_gradient_hierarchy.data_ptrs[layer_id],
												  NULL,
												  SLAM_system_settings::instance()->sensor_params,
												  this->cache_increment_pose,
												  layer_id,
												  this->dev_accumulation_buffer);
				//CUDA_CKECK_KERNEL;

				// Copy out result 
				checkCudaErrors(cudaMemcpy((void *)&this->photometric_accumulation, this->dev_accumulation_buffer, sizeof(Accumulate_result),
					cudaMemcpyDeviceToHost));
				copy_out_hessian_from_accumulate_data(photometric_hessian, photometric_nabla, this->photometric_accumulation);
				// Normalize residual
				if (this->photometric_accumulation.number_of_pairs > 100)
				{
					photometric_hessian /= this->photometric_accumulation.number_of_pairs;
					photometric_nabla /= this->photometric_accumulation.number_of_pairs;
				}
				else
				{
					photometric_hessian.setIdentity();
					photometric_hessian.setZero();
				}
			}


			// ----- Residual computation (Keypoints --> Hessian, Nabla)
			{
				//
				keypoint_hessian.setZero();
				keypoint_nabla.setZero();
				//
				int number_of_keypoints = current_keypoints.size();
				Eigen::Matrix4f cache_increment_pose_mat(this->cache_increment_pose.data);
				//// Precompute medium residual
				//float mean_value = 0.0f;
				//int mean_counter = 0;
				//for (int point_id = 0; point_id < number_of_keypoints; point_id++)
				//{
				//	Eigen::Vector3f current_point(current_keypoints[point_id]);
				//	Eigen::Vector3f model_point(model_keypoints[point_id]);
				//	if (model_point.z() < 0.01)	continue;
				//	current_point = cache_increment_pose_mat.block(0, 0, 3, 3) * current_point.eval() + cache_increment_pose_mat.block(0, 3, 3, 1);
				//	mean_value += (current_point - model_point).norm();
				//	mean_counter++;
				//}
				//mean_value /= (float)mean_counter;

				// Compute medium residual
				int residual_counter = 0;
				for (int point_id = 0; point_id < number_of_keypoints; point_id++)
				{
					Eigen::Vector3f current_point(current_keypoints[point_id]);
					Eigen::Vector3f model_point(model_keypoints[point_id]);
					if (model_point.z() < 0.01)	continue;

					current_point = cache_increment_pose_mat.block(0, 0, 3, 3) * current_point.eval() + cache_increment_pose_mat.block(0, 3, 3, 1);
					
					//
					Eigen::MatrixXf J_mat(3, 6), b_mat(3, 1);
					J_mat.setZero();
					J_mat.block(0, 3, 3, 3).setIdentity();
					//
					J_mat.data()[1] = -current_point.z();	J_mat.data()[2] = +current_point.y();	J_mat.data()[6] = -current_point.y();
					J_mat.data()[3] = +current_point.z();	J_mat.data()[5] = -current_point.x();	J_mat.data()[7] = +current_point.x();
					//
					b_mat = current_point - model_point;


					//
					float point_diff = (current_point - model_point).norm();
					//float huber_radius = mean_value * 3;
					float huber_radius = this->keypoint_huber;
					if (point_diff < huber_radius)
					{
						keypoint_hessian += J_mat.transpose() * J_mat;
						keypoint_nabla += J_mat.transpose() * b_mat;
						residual_counter++;
					}
					else if (point_diff < huber_radius * 3)
					{
						//keypoint_hessian += 0;
						Eigen::MatrixXf clamp_b_mat(3, 1);
						for (int i = 0; i < 3; i++)
							clamp_b_mat.data()[i] = fmaxf(fminf(b_mat.data()[i], this->keypoint_huber), -this->keypoint_huber);
						keypoint_nabla += J_mat.transpose() * clamp_b_mat;
						residual_counter++;
					}
					else
					{
						/* Invalid residual */
					}
				}
				// Normalize residual
				//printf("%d\n", residual_counter);
				if (residual_counter > 10)
					//if (residual_counter > 50)
				{
					keypoint_hessian /= residual_counter;
					keypoint_nabla /= residual_counter;
				}
				else
				{
					keypoint_hessian.setIdentity();
					keypoint_nabla.setZero();
					// Don't use keypoint information
					keypoint_weight = 0.0;
				}
			}

			// ----- Validate number of point pairs
			if (this->icp_accumulation.number_of_pairs <= 10)
			{
				return TrackingState::TRACKING_FAILED;
			}
			// ----- Solver (Hessian, Nabla --> X=[rx,ry,rz,tx,ty,tz] --> increment pose matrix)
Solve_one_iteration :
			{
				this->total_hessian = icp_weight * icp_hessian + photometric_weight * photometric_weight * photometric_hessian +
					keypoint_weight * keypoint_hessian;
				this->total_nabla = -(icp_weight * icp_nabla + photometric_weight * photometric_nabla + keypoint_weight * keypoint_nabla);
				//this->total_hessian = keypoint_hessian;
				//this->total_nabla = -(keypoint_nabla);

				//printf("norm = %f\n", this->total_hessian.lpNorm<2>());

				// Compute average point distance
				float average_distance = this->icp_accumulation.energy / (float)this->icp_accumulation.number_of_pairs;

				//IterationState state = my_solver(this->total_hessian, this->total_nabla, this->incremental_pose);
				state = my_solver(this->total_hessian, this->total_nabla,
												 average_distance, this->icp_accumulation.number_of_pairs,
												 this->incremental_pose);
				if (state == IterationState::FAILED || state == IterationState::CONVERGENCED)	break;
			}
		}
	}

	if (state == IterationState::FAILED)
	{
		camera_pose = camera_pose.eval();
		printf("Bad Tracking\n");
	}
	else
	{
		//
		camera_pose = camera_pose.eval() * this->incremental_pose;
	}

	return TrackingState::TRACKINE_SUCCED;
}




//
void Keypoint_ICP_tracker::prepare_to_track_new_frame()
{
	// Set tracking state
	this->tracking_state = TrackingState::BEFORE_TRACKING;
	this->incremental_pose.setIdentity();
}

//
void Keypoint_ICP_tracker::prepare_to_new_iteration()
{
	//
	this->total_hessian.setZero();
	this->total_nabla.setZero();

	// 
	checkCudaErrors(cudaMemset(this->dev_accumulation_buffer, 0x00, sizeof(Accumulate_result)));

	//
	memcpy(cache_increment_pose.data, this->incremental_pose.data(), 4 * 4 * sizeof(float));
}

#pragma endregion
