


#include "math.h"
#include "Map_engine/Feature_map.h"



//
Feature_map::Feature_map()
{
	scale_layer_depth[0] = FEATURE_SCALE_FACTOR;
	for (int i = 1; i < MAX_LAYER_NUMBER; i++)
		scale_layer_depth[i] = scale_layer_depth[i - 1] * FEATURE_SCALE_FACTOR;
}
Feature_map::~Feature_map()
{
}


//
void Feature_map::update_current_features(std::vector<My_Type::Vector3f> current_keypoints,
										  cv::Mat current_features,
										  std::vector<int> & current_match_to_model,
										  std::vector<int> & previous_match_to_model,
										  My_pose & camera_pose)
{
	int number_of_model_keypoint = this->model_keypoints.size();
	

	// Update keypoint and feature
	for (int current_id = 0; current_id < current_keypoints.size(); current_id++)
	{
		int match_model_id = current_match_to_model[current_id];

		// Current keypoint in map coordinate
		Eigen::Vector4f current_vec_in_map(current_keypoints[current_id].x,
										   current_keypoints[current_id].y,
										   current_keypoints[current_id].z, 1.0f);
		current_vec_in_map = camera_pose.mat * current_vec_in_map.eval();

		// Compute scale
		int min_scale_id;
		float min_diff_value = FLT_MAX;
		for (int scale_id = 0; scale_id < MAX_LAYER_NUMBER; scale_id++)
		{
			float scale_diff = fabsf(scale_layer_depth[scale_id] - current_keypoints[current_id].z);
			if (min_diff_value > scale_diff)
			{	min_scale_id = scale_id;	min_diff_value = scale_diff;	}
		}
		//
		cv::Mat new_feature = current_features.row(current_id);
		//printf("%d\n", this->model_keypoints.size());
		// Update
		if (match_model_id >= number_of_model_keypoint)
		{
			// New keypoint 
			// - Keypoint / weight / tracking count
			Model_keypoint new_keypoint;
			new_keypoint.point = My_Type::Vector3f(current_vec_in_map.x(), current_vec_in_map.y(), current_vec_in_map.z());
			new_keypoint.weight = 1.0f;
			new_keypoint.is_valid = true;
			new_keypoint.non_observe_counter = 0;
			new_keypoint.observe_counter = 1;
			this->model_keypoints.push_back(new_keypoint);
			// - Feature
			Scale_layer_flag new_scale_layer_flag;
			new_scale_layer_flag.exist[min_scale_id] = 1;
			cv::Mat new_multiscale_feature;
			new_multiscale_feature.create(cv::Size(new_feature.cols, MAX_LAYER_NUMBER), new_feature.type());
			new_feature.copyTo(new_multiscale_feature.row(min_scale_id));
			if (min_scale_id > 0)
			{
				new_feature.copyTo(new_multiscale_feature.row(min_scale_id - 1));
				new_scale_layer_flag.exist[min_scale_id - 1] = 1;
			}
			if (min_scale_id < MAX_LAYER_NUMBER - 1)
			{
				new_feature.copyTo(new_multiscale_feature.row(min_scale_id + 1));
				new_scale_layer_flag.exist[min_scale_id + 1] = 1;
			}
			this->model_features.push_back(new_multiscale_feature);
			this->model_feature_scale_flag.push_back(new_scale_layer_flag);


			// - Mapper
			My_Type::Vector3i keypoint_block_pos(new_keypoint.point.x / FEATRUE_BLOCK_WIDTH,
												 new_keypoint.point.y / FEATRUE_BLOCK_WIDTH,
												 new_keypoint.point.z / FEATRUE_BLOCK_WIDTH);
			this->map_point_mapper[keypoint_block_pos] = match_model_id;
			//
			current_match_to_model[current_id] = match_model_id;
			//printf("%d, %d\n", this->model_keypoints.size(), match_model_id);
		}
		else if (match_model_id >= 0)
		{
			// Existed keypoint
			// - Update position / weight / tracking count
			My_Type::Vector3f model_keypoint_position(this->model_keypoints[match_model_id].point.data);
			// Check position
			if ((model_keypoint_position - My_Type::Vector3f(current_vec_in_map.x(), current_vec_in_map.y(), current_vec_in_map.z())).norm() > 0.2)
			{
				current_match_to_model[current_id] = -1;
				continue;
			}
			float model_weight = this->model_keypoints[match_model_id].weight;
			this->model_keypoints[match_model_id].point = (model_keypoint_position * model_weight +
														   My_Type::Vector3f(current_vec_in_map.x(), current_vec_in_map.y(), current_vec_in_map.z())) /
														   (model_weight + 1.0f);
			if (this->model_keypoints[match_model_id].weight < 16.0f)
				this->model_keypoints[match_model_id].weight += 1.0f;
			this->model_keypoints[match_model_id].observe_counter++;
			// - Update feature
			// To Do new_feature.copyTo(new_multiscale_feature.row(min_scale_id));
			new_feature.copyTo(this->model_features[match_model_id].row(min_scale_id));
			if (min_scale_id > 0)
			{
				new_feature.copyTo(this->model_features[match_model_id].row(min_scale_id - 1));
				this->model_feature_scale_flag[match_model_id].exist[min_scale_id - 1] ++;
			}
			if (min_scale_id < MAX_LAYER_NUMBER - 1)
			{
				new_feature.copyTo(this->model_features[match_model_id].row(min_scale_id + 1));
				this->model_feature_scale_flag[match_model_id].exist[min_scale_id + 1] ++;
			}
			this->model_feature_scale_flag[match_model_id].exist[min_scale_id]++;
		}
		else
		{
			/* Invalid keypoint */
			current_match_to_model[current_id] = -1;
		}
	}
	previous_match_to_model = current_match_to_model;

	// Update model keypoint status
	for (int model_id = 0; model_id < this->model_keypoints.size(); model_id++)
	{
		if (this->model_keypoints[model_id].is_valid && 
			this->model_keypoints[model_id].observe_counter < this->valid_observe_count)
		{
			this->model_keypoints[model_id].non_observe_counter++;
			if (this->model_keypoints[model_id].non_observe_counter > this->outlyer_non_observe_count)
				this->model_keypoints[model_id].is_valid = false;
		}
	}
}


//
void Feature_map::save_keyframe(cv::Mat current_features,
								std::vector<int> current_match_to_model, 
								DBoW3::Vocabulary & feature_voc)
{
	if (current_features.rows < 10)	return;

	//
	this->keyframe_feature_mapper_list.push_back(current_match_to_model);
	
	// DBoW add frame to database
	DBoW3::BowVector current_feature_vec;
	// Transfrom to pre-trained vocabulary space
	feature_voc.transform(current_features, current_feature_vec);
	this->dbow_vec_list.push_back(current_feature_vec);


	// Compute weight center of keypoints
	My_Type::Vector3f accu_buffer(0.0f, 0.0f, 0.0f);
	int valid_counter = 0;
	for (int point_id = 0; point_id < current_match_to_model.size(); point_id++)
	{ 
		int model_keypoint_id = current_match_to_model[point_id];
		if (model_keypoint_id < 0)	continue;

		My_Type::Vector3f temp_point = this->model_keypoints[model_keypoint_id].point;
		accu_buffer += temp_point;
		valid_counter++;
	}
	accu_buffer /= valid_counter;
	//
	this->keyframe_weigth_centers.push_back(accu_buffer);
}


//
void Feature_map::get_model_keypoints(std::vector<My_Type::Vector3f> & current_keypoints,
									  My_pose & camera_pose,
									  std::vector<std::vector<My_Type::Vector3f>> & visible_keypoints,
									  std::vector<cv::Mat> & visible_features,
									  std::vector<std::vector<int>> & visible_point_model_index)
{
	visible_keypoints.clear();
	visible_point_model_index.clear();
	visible_features.clear();

	// 
	for (int current_id = 0; current_id < current_keypoints.size(); current_id++)
	{
		// Compute scale
		int min_scale_id;
		float min_diff_value = FLT_MAX;
		for (int scale_id = 0; scale_id < MAX_LAYER_NUMBER; scale_id++)
		{
			float scale_diff = fabsf(scale_layer_depth[scale_id] - current_keypoints[current_id].z);
			if (min_diff_value > scale_diff)
			{
				min_scale_id = scale_id;	min_diff_value = scale_diff;
			}
		}

		// Transfer to map coordinate 
		Eigen::Vector4f current_vec_in_map(current_keypoints[current_id].x,
										   current_keypoints[current_id].y,
										   current_keypoints[current_id].z, 1.0f);
		current_vec_in_map = camera_pose.mat * current_vec_in_map.eval();

		//
		float search_radius_3D = this->search_projection_radius / SLAM_system_settings::instance()->sensor_params.sensor_fx * current_keypoints[current_id].z;
		My_Type::Vector3f base_point(current_vec_in_map.x(), current_vec_in_map.y(), current_vec_in_map.z());
		base_point = base_point - My_Type::Vector3f(search_radius_3D, search_radius_3D, search_radius_3D);
		int number_of_step = ceilf(search_radius_3D * 2 / FEATRUE_BLOCK_WIDTH);
		number_of_step = std::max(number_of_step, 2);
		// 
		std::vector<My_Type::Vector3f> neighbor_points;
		cv::Mat neighbor_feature;
		std::vector<int> neighbor_point_index;
		// Search neighbor keypoint
		for (int step_z = 0; step_z < number_of_step; step_z++)	
			for (int step_y = 0; step_y < number_of_step; step_y++)	
				for (int step_x = 0; step_x < number_of_step; step_x++)
		{
			My_Type::Vector3i check_vec((int)(base_point.x / FEATRUE_BLOCK_WIDTH) + step_x,
										(int)(base_point.y / FEATRUE_BLOCK_WIDTH) + step_y,
										(int)(base_point.z / FEATRUE_BLOCK_WIDTH) + step_z);
			// Search point
			std::unordered_map<My_Type::Vector3i, int>::iterator point_index_iter = this->map_point_mapper.find(check_vec);
			if (point_index_iter != this->map_point_mapper.end())
			{
				My_Type::Vector3f model_point_vec;
				int model_point_index = (*point_index_iter).second;

				//
				if (!this->model_keypoints[model_point_index].is_valid)	continue;
				// Get keypoint position
				model_point_vec = this->model_keypoints[model_point_index].point;
				// Push back keypoints and indexes
				neighbor_points.push_back(model_point_vec);
				neighbor_point_index.push_back(model_point_index);
				neighbor_feature.push_back(this->model_features[model_point_index].row(min_scale_id).clone());
			}
		}
		
		// 
		visible_keypoints.push_back(neighbor_points);
		visible_point_model_index.push_back(neighbor_point_index);
		visible_features.push_back(neighbor_feature.clone());
	}
}


