#pragma once

//
#include "UI_engine/UI_parameters.h"
#include "SLAM_system/SLAM_system_settings.h"
#include "Preprocess_engine/Hierarchy_image.h"
#include "OurLib/Trajectory_node.h"
//
#include "Plane_detector/Plane_detector.h"
#include "Map_engine/voxel_definition.h"
//#include "Map_engine/Plane_map.h"
//#include "Map_engine/Voxel_map.h"
//


//!
enum MainViewportRenderMode
{
	PHONG_RENDER = 0,
	SDF_WEIGHT_RENDER = 1,
	SEMANTIC_PLANE_RENDER = 2,
};


//!
enum NormalsSource
{
	DEPTH_NORMAL = 0,
	MODEL_NORMAL = 1,
};

//!
/*!


*/
class Render_engine
{
public:

	//! 
	My_Type::Vector2i depth_size;
	//! 
	My_Type::Vector2i scene_depth_size;
	//!
	My_Type::Vector2i range_map_size;


	//! Ground truth trajectory
	Trajectory gound_truth_trajectory;
	//! Estimated trajectory
	Trajectory estiamted_trajectory;


	//! Current frame points
	My_Type::Vector3f * current_points;
	//! Model points (generate by raycast_module)
	My_Type::Vector3f * model_points;


	
	//! Current hierarchy normals 
	Hierarchy_image<My_Type::Line_segment> current_hierarchy_normal_to_draw;
	//! Model hierarchy normals
	Hierarchy_image<My_Type::Line_segment> model_hierarchy_normal_to_draw;
	//! Current hierarchy normals 
	Hierarchy_image<My_Type::Line_segment> dev_current_hierarchy_normal_to_draw;
	//! Model hierarchy normals
	Hierarchy_image<My_Type::Line_segment> dev_model_hierarchy_normal_to_draw;


	//! Raycast range map (line segment)
	My_Type::Vector2f * range_map;

	//! Points for OpenGL rendering
	My_Type::Vector3f * scene_points, *dev_scene_points;
	//! Normal vector of scene_points
	My_Type::Vector3f * scene_normals, *dev_scene_normals;
	//! Normal vector of scene_points
	My_Type::Vector3f * dev_scene_plane_label;
	//! Weight of scene_points
	int * dev_scene_points_weight;
	//! Color array of scene_points
	My_Type::Vector4uc * scene_points_color, * dev_scene_points_color;


	//! Pseudo color array of plane labels
	My_Type::Vector4uc * pseudo_plane_color, * dev_pseudo_plane_color;

	//!
	HashEntry * enties_buffer;
	//!
	int number_of_blocks = 0;
	My_Type::Vector3f * voxel_block_lines;


#pragma region(Viewport2)

	//! 
	int min_depth, * dev_min_depth;	/* millimeter */
	//!
	int max_depth, * dev_max_depth;	/* millimeter */
	//! Color buffer of viewport 2
	My_Type::Vector4uc * viewport_2_color, * dev_viewport_2_color;

#pragma endregion



#pragma region(Pixel/Voxel block position)
	


#pragma endregion


	//!
	Render_engine();
	~Render_engine();


	//! Initialization
	/*!
		\param	depth_size			Size of depth image

		\param	scene_depth_size	Size of depth image

		\return	void
	*/
	void init(My_Type::Vector2i depth_size, 
			  My_Type::Vector2i scene_depth_size);


	//!
	/*!
		\param	scene_depth_size	Size of depth image size

		\return	void
	*/
	void scene_viewport_reshape(My_Type::Vector2i scene_depth_size);


	//!
	/*!
		\param	render_mode		Render mode of scene points

		\return	void
	*/
	void render_scene_points(MainViewportRenderMode render_mode);


	//!
	/*!
	
	*/
	void pseudo_render_depth(My_Type::Vector3f * dev_raw_aligned_points);


	//!
	/*!
	
	*/
	void pseudo_render_plane_labels(int * dev_plane_labels);


	//!
	/*!
	
	*/
	void generate_normal_segment_line(My_Type::Vector3f * dev_raw_aligned_points,
									  My_Type::Vector3f * dev_normals,
									  NormalsSource normals_source);

	//!
	/*!
	
	*/
	void generate_voxel_block_lines(HashEntry * dev_entries, 
									int number_of_entries = (ORDERED_TABLE_LENGTH + EXCESS_TABLE_LENGTH));


};

