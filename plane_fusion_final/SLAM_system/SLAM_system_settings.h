#pragma once

//
#include "OurLib/My_matrix.h"


//
//#define CUDA_CKECK_KERNEL
#define CUDA_CKECK_KERNEL		checkCudaErrors(cuCtxSynchronize());

// Enable debug code 
#define COMPILE_DEBUG_CODE



#ifndef MAX_LAYER_NUMBER
#define MAX_LAYER_NUMBER	8
#endif


//!
typedef struct _Sensor_params
{
	//! Fx, Fy
	float sensor_fx, sensor_fy;
	//! Cx, Cy
	float sensor_cx, sensor_cy;
	//! Sensor scale (For example: Kinect scale = 1.0m / 1.0mm = 1000.0f)
	float sensor_scale;
	//! Sensor noise radius per meter
	float sensor_noise_ratio;
	//! Sensor range
	float min_range, max_range;
}Sensor_params;


//!
/*!

*/
class SLAM_system_settings
{
public:
	//! The pointer to this static object.
	static SLAM_system_settings * instance_ptr;
	//! Member function for instantiating this static object.
	static SLAM_system_settings * instance(void)
	{
		if (instance_ptr == nullptr)	instance_ptr = new SLAM_system_settings();
		return instance_ptr;
	}

	//! Generate mesh for better visualization
	bool generate_mesh_for_visualization;
	//! Use plane information
	bool enable_plane_module;

#pragma region(Image/sensor parameters)
	//! CUDA memory alignment size (must be setted to {1, 2, 4, ..., 16})
	int image_alginment_patch_width;
	//! Raycast range patch size (no bigger than this->image_alginment_patch_width!)
	int raycast_range_patch_width;

	//! Aligned depth image size 
	My_Type::Vector2i aligned_depth_size;
	//! Aligned color image size 
	My_Type::Vector2i aligned_color_size;

	//! Sensor parameters
	Sensor_params sensor_params;
    Sensor_params color_params;
    Sensor_params depth_params;
    My_Type::Matrix44f depth2color_mat, depth2imu_mat;
#pragma endregion


#pragma region(Plane detector parameters)
	//! Pre-segment cell width (NOTE: Must be the common divisor of aligned_depth_size!)
	int presegment_cell_width;

	//! Weight coefficients of super pixel cluster
	float pixel_data_weight;
	float normal_position_weight;


#pragma endregion


#pragma region(Tracker parameters)
	//!
	int number_of_hierarchy_layers;

	//! Max iterate times of each layer
	int max_iterate_times[MAX_LAYER_NUMBER];

	//! Convergence threshold of nonlinear solver (Tracker)
	float convergence_threshold;
	//! Failed threshold of nonlinear solver (Tracker)
	float failure_threshold;

#pragma endregion

	//!
	SLAM_system_settings();
	~SLAM_system_settings();

	//! Set all paramenters to default value.
	void set_to_default();
    void set_intrinsic(const float & fx, const float & fy,
                       const float & cx, const float & cy);
	//! Set depth image size
	void set_depth_image_size(const int & width, const int & height, bool align_flag = true);
	//! Set color image size 
	void set_color_image_size(const int & width, const int & height, bool align_flag = true);

	//! Set calibration paramters
	void set_calibration_paramters(const float & fx, const float & fy, 
								   const float & cx, const float & cy, 
								   const float & scale);
    void set_intrinsic(const float & fx, const float & fy,
                       const float & cx, const float & cy,
                       const float & scale);

    void set_extrinsic(const My_Type::Matrix44f d2c, const My_Type::Matrix44f d2i);

};

