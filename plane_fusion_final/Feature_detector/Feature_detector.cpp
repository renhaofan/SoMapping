

#include "Feature_detector/Feature_detector.h"
#include "SLAM_system/SLAM_system_settings.h"


// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

//
#include "Feature_detector/Feature_detector_KernelFunc.cuh"


//
std::vector<float> compute_response(const cv::Mat& img, std::vector<cv::Point2f>& pts, int blockSize);



Feature_detector::Feature_detector()
{
	this->dev_gray_image_buffer = nullptr;
	//
	this->orb_feature_extractor = new ORB_SLAM2::ORBextractor(this->aim_number_of_features,
															  this->orb_scale_factor,
															  this->number_of_feature_levels,
															  this->iniThFAST,
															  this->minThFAST);
	//
	scale_layer_depth[0] = orb_scale_factor;
	for (int i = 1; i < MAX_LAYER_NUMBER; i++)
		scale_layer_depth[i] *= scale_layer_depth[i - 1];


	//
	sdkCreateTimer(&this->timer_average);
	sdkResetTimer(&this->timer_average);
}
Feature_detector::~Feature_detector()
{
	checkCudaErrors(cudaFree(this->dev_gray_image_buffer));
}


//
void Feature_detector::prepare_to_detect_features()
{
	this->current_keypoints.clear();
	this->current_features.release();
	this->current_keypoint_position.clear();
}


//
void Feature_detector::detect_orb_features(Hierarchy_image<float> & dev_current_intensity,
										   Hierarchy_image<My_Type::Vector3f> & current_points)
{
	// Prepare to detect features
	{
		//
		this->prepare_to_detect_features();
		//
		My_Type::Vector2i image_size(dev_current_intensity.size[0].width, dev_current_intensity.size[0].height);
		My_Type::Vector2i buffer_size(this->gray_image.cols, this->gray_image.rows);
		//
		if (image_size != buffer_size)
		{
			if (!this->gray_image.empty())	this->gray_image.release();
			this->gray_image.create(image_size.height, image_size.width, CV_8UC1);

			if (this->dev_gray_image_buffer)
			{
				checkCudaErrors(cudaFree(this->dev_gray_image_buffer));
				this->dev_gray_image_buffer = nullptr;
			}
			checkCudaErrors(cudaMalloc((void **)&(this->dev_gray_image_buffer),
				image_size.width * image_size.height * sizeof(unsigned char)));
		}


		//
		dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);
		thread_rect.x = SLAM_system_settings::instance()->image_alginment_patch_width;
		thread_rect.y = SLAM_system_settings::instance()->image_alginment_patch_width;
		thread_rect.z = 1;
		block_rect.x = image_size.width / thread_rect.x;
		block_rect.y = image_size.height / thread_rect.y;
		block_rect.z = 1;
		convert_intensity_to_cvmat_KernelFunc(block_rect, thread_rect,
											  dev_current_intensity.data_ptrs[0], image_size,
											  My_Type::Vector2f(0.0f, 1.0f),
											  this->dev_gray_image_buffer);
		//CUDA_CKECK_KERNEL;

		checkCudaErrors(cudaMemcpy(this->gray_image.data, this->dev_gray_image_buffer,
			image_size.width * image_size.height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	}
	
	if (false)
	{
		/* not use */
		//Detect ORB features (use ORB-SLAM2::ORBextractor)
		(*this->orb_feature_extractor)(this->gray_image, cv::Mat(),
									   this->current_keypoints, this->current_features);
	}
	else if (true)
	{

		// init
		this->current_match_to_model_id.clear();
		this->current_keypoints_2d.clear();
		this->current_match_to_model_id.reserve(this->previous_match_to_model_id.size());
		this->current_keypoints_2d.reserve(this->previous_match_to_model_id.size());

		// Match keypoint by LK optic flow
		std::vector<int> current_match_to_previous;
		current_match_to_previous.resize(this->current_keypoints.size());
		if (!this->previous_gray_image.empty())
		{
			std::vector<unsigned char> track_status;
			std::vector<cv::Point2f> tracked_keypoints_2d;
			cv::calcOpticalFlowPyrLK(this->previous_gray_image, this->gray_image,
									 this->previous_keypoints_2d, tracked_keypoints_2d,
									 track_status, cv::noArray());
			//
			for (int previous_id = 0; previous_id < tracked_keypoints_2d.size(); previous_id++)
			{
				if (track_status[previous_id] == 1 &&
					tracked_keypoints_2d[previous_id].x > 16 && 
					tracked_keypoints_2d[previous_id].x <= this->gray_image.cols - 16 &&
					tracked_keypoints_2d[previous_id].y > 16 && 
					tracked_keypoints_2d[previous_id].y <= this->gray_image.rows - 16 &&
					this->previous_match_to_model_id[previous_id] >= 0 &&
					previous_id < this->previous_match_to_model_id.size())
				{
					// Match id 
					this->current_match_to_model_id.push_back(this->previous_match_to_model_id[previous_id]);
					// Keypoint-2d
					this->current_keypoints_2d.push_back(tracked_keypoints_2d[previous_id]);
				}
			}


			// Eliminate weak keypoints
			float response_threshold;
			std::vector<float> response_vector = compute_response(this->gray_image, this->current_keypoints_2d, 3);
			std::vector<float> response_vector_to_sort(response_vector);
			std::sort(response_vector_to_sort.begin(), response_vector_to_sort.end());
			if (response_vector_to_sort.size() > this->aim_number_of_features)
			{	response_threshold = response_vector_to_sort[response_vector_to_sort.size() - this->aim_number_of_features];	}
			else if (response_vector_to_sort.size() > 0)
			{	response_threshold = response_vector_to_sort.front();	}
			// 
			std::vector<int> keypoint_match_id_buffer;
			std::vector<cv::Point2f> keypoint_2d_buffer;
			for (int response_id = 0; response_id < response_vector.size(); response_id++)
			{
				if (response_vector[response_id] >= response_threshold)
				{
					keypoint_match_id_buffer.push_back(this->current_match_to_model_id[response_id]);
					keypoint_2d_buffer.push_back(this->current_keypoints_2d[response_id]);
				}
			}
			this->current_match_to_model_id = keypoint_match_id_buffer;
			this->current_keypoints_2d = keypoint_2d_buffer;

			// Debug
			//printf("%f\n", (float)this->current_keypoints_2d.size() / (float)tracked_keypoints_2d.size());

		}

		//sdkResetTimer(&(this->timer_average));
		//sdkStartTimer(&(this->timer_average));
		//sdkStopTimer(&(this->timer_average));
		//float elapsed_time = sdkGetAverageTimerValue(&(this->timer_average));
		////float elapsed_time = sdkGetTimerValue(&(this->timer_average));
		//printf("%f\n", elapsed_time);


		// Good feature to track detect keypoints
		std::vector<cv::Point2f> new_detected_keypoints_2d;
		bool use_mask = false;
		if (use_mask)
		{
			// Generate mask
			this->detected_mask = 255 * cv::Mat::ones(this->gray_image.size(), this->gray_image.type());
			for (int tracked_kp_id = 0; tracked_kp_id < this->current_keypoints_2d.size(); tracked_kp_id++)
			{
				int u_base = (int)this->current_keypoints_2d[tracked_kp_id].x;
				int v_base = (int)this->current_keypoints_2d[tracked_kp_id].y;
				const int multiple_value = 1;
				for (int u_offset = -this->keypoint_window_radius * multiple_value; u_offset < this->keypoint_window_radius * multiple_value; u_offset++)
					for (int v_offset = -this->keypoint_window_radius * multiple_value; v_offset < this->keypoint_window_radius * multiple_value; v_offset++)
					{
						int u_check = u_base + u_offset;
						int v_check = v_base + v_offset;
						if (u_check < 0 || u_check >= this->detected_mask.cols ||
							v_check < 0 || v_check >= this->detected_mask.rows)		continue;
						// 
						this->detected_mask.data[u_check + v_check * this->detected_mask.cols] = 0;
					}
			}
			//cv::imshow("this->detected_mask", this->detected_mask); 
			cv::goodFeaturesToTrack(this->gray_image, new_detected_keypoints_2d,
									(int)this->aim_number_of_features / 3,
									(double)0.01, (double)this->keypoint_window_radius, this->detected_mask, 3);
		}
		else
		{
			std::vector<cv::Point2f> new_keypoint_swap_buffer;
			cv::goodFeaturesToTrack(this->gray_image, new_keypoint_swap_buffer,
									(int)this->aim_number_of_features / 3,
									(double)0.01, (double)this->keypoint_window_radius, cv::noArray(), 3);
			//
			this->detected_mask = 255 * cv::Mat::ones(this->gray_image.size(), CV_8UC1);
			for (int tracked_kp_id = 0; tracked_kp_id < this->current_keypoints_2d.size(); tracked_kp_id++)
			{
				int u_base = (int)this->current_keypoints_2d[tracked_kp_id].x;
				int v_base = (int)this->current_keypoints_2d[tracked_kp_id].y;
				const int multiple_value = 1;
				for (int u_offset = -this->keypoint_window_radius * multiple_value; u_offset < this->keypoint_window_radius * multiple_value; u_offset++)
					for (int v_offset = -this->keypoint_window_radius * multiple_value; v_offset < this->keypoint_window_radius * multiple_value; v_offset++)
					{
						int u_check = u_base + u_offset;
						int v_check = v_base + v_offset;
						if (u_check < 0 || u_check >= this->detected_mask.cols ||
							v_check < 0 || v_check >= this->detected_mask.rows)		continue;
						// 
						this->detected_mask.data[u_check + v_check * this->detected_mask.cols] = 0;
					}
			}
			//cv::imshow("this->detected_mask", this->detected_mask); 
			//
			for (int new_keypoint_id = 0; new_keypoint_id < new_keypoint_swap_buffer.size(); new_keypoint_id++)
			{
				int mask_index = (int)new_keypoint_swap_buffer[new_keypoint_id].x + (int)new_keypoint_swap_buffer[new_keypoint_id].y * this->detected_mask.cols;
				if (this->detected_mask.data[mask_index] != 0 &&
					new_keypoint_swap_buffer[new_keypoint_id].x > 16 &&
					new_keypoint_swap_buffer[new_keypoint_id].x <= this->gray_image.cols - 16 &&
					new_keypoint_swap_buffer[new_keypoint_id].y > 16 &&
					new_keypoint_swap_buffer[new_keypoint_id].y <= this->gray_image.rows - 16)
				{
					new_detected_keypoints_2d.push_back(new_keypoint_swap_buffer[new_keypoint_id]);
				}
			}
		}
		//
		this->current_keypoints_2d.insert(this->current_keypoints_2d.end(), 
										  new_detected_keypoints_2d.begin(), new_detected_keypoints_2d.end());
		//
		std::vector<int> temp_match_id(new_detected_keypoints_2d.size(), -1);
		this->number_of_tracked_keypoints = this->current_match_to_model_id.size();
		this->current_match_to_model_id.insert(this->current_match_to_model_id.end(),
											   temp_match_id.begin(), temp_match_id.end());


		// Compute ORB features
		this->current_keypoints.clear();
		for (int pt_id = 0; pt_id < this->current_keypoints_2d.size(); pt_id++)
			this->current_keypoints.push_back(cv::KeyPoint(this->current_keypoints_2d[pt_id], 3));
		cv::Ptr<cv::ORB> orb_detector_ptr = cv::ORB::create(this->aim_number_of_features);
		// 
		std::vector<cv::KeyPoint> compare_buffer = this->current_keypoints;
		orb_detector_ptr->compute(this->gray_image, this->current_keypoints, this->current_features);
		//
		std::vector<int> match_id_buffer;
		for (int compare_id = 0, new_id = 0; compare_id < compare_buffer.size(); compare_id++)
		{
			if (new_id >= this->current_keypoints.size())	break;

			if (compare_buffer[compare_id].pt == this->current_keypoints[new_id].pt)
			{
				match_id_buffer.push_back(this->current_match_to_model_id[compare_id]);
				new_id++;
			}
		}
		this->current_match_to_model_id = match_id_buffer;

		//printf("1");
	}
	else
	{
		/* not use */
		cv::Ptr<cv::ORB> orb_detector_ptr = cv::ORB::create(this->aim_number_of_features);
		orb_detector_ptr->detect(this->gray_image, this->current_keypoints);
		orb_detector_ptr->compute(this->gray_image, this->current_keypoints, this->current_features);
	}


	// Show detected features
	if (false)
	{
		cv::Mat draw_buffer;
		drawKeypoints(this->gray_image, this->current_keypoints,
					  draw_buffer, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
		cv::imshow("1", draw_buffer);
	}

	// Init 3D position for each keypoint (get from depth)
	this->current_keypoint_position.resize(this->current_keypoints.size());
	for (int current_id = 0; current_id < this->current_keypoints.size(); current_id++)
	{
		int u = this->current_keypoints_2d[current_id].x;
		int v = this->current_keypoints_2d[current_id].y;
		int image_width = SLAM_system_settings::instance()->aligned_depth_size.width;

		//

		//
		float min_z_point = FLT_MAX;
		const int window_radius = 2;
		for (int v_inc = -window_radius; v_inc <= window_radius; v_inc++)
			for (int u_inc = -window_radius; u_inc <= window_radius; u_inc++)
		{
			int u_check = u + v_inc;
			int v_check = v + v_inc;
			if (u_check < window_radius && u_check >= this->gray_image.cols - window_radius &&
				v_check < window_radius && v_check >= this->gray_image.rows - window_radius)	continue;

			int point_index = u_check + v_check * image_width;
			if (current_points.data_ptrs[0][point_index].z > 0.1f)
				min_z_point = fminf(min_z_point, current_points.data_ptrs[0][point_index].z);
		}
		//
		float min_distance_to_center = FLT_MAX;
		My_Type::Vector3f temp_vec3f(0.0f, 0.0f, 0.0f);
		for (int v_inc = -window_radius; v_inc <= window_radius; v_inc++)
			for (int u_inc = -window_radius; u_inc <= window_radius; u_inc++)
		{
			int u_check = u + v_inc;
			int v_check = v + v_inc;
			if (u_check < window_radius && u_check >= this->gray_image.cols - window_radius &&
				v_check < window_radius && v_check >= this->gray_image.rows - window_radius)	continue;

			int point_index = u_check + v_check * image_width;
			if (fabsf(current_points.data_ptrs[0][point_index].z - min_z_point) > 0.05f)	continue;

			float distance_to_center = v_inc + u_inc;
			if (distance_to_center < min_distance_to_center)
			{
				min_distance_to_center = distance_to_center;
				temp_vec3f = current_points.data_ptrs[0][point_index];
			}
		}

		//
		this->current_keypoint_position[current_id] = temp_vec3f;
	}


	// Update previous gray image
	this->previous_gray_image = this->gray_image.clone();
	//// Update previous matches (update in function update_map())
	//this->previous_match_to_model_id = this->current_match_to_model_id;
	// Update previous keypoint-2d
	this->previous_keypoints_2d = this->current_keypoints_2d;


}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
	const int *pa = a.ptr<int>();
	const int *pb = b.ptr<int>();

	int dist = 0;

	for (int i = 0; i < 8; i++, pa++, pb++)
	{
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

//
void Feature_detector::match_orb_features(int number_of_model_keypoints)
{

	// Match keypoint by feature distance (match to neighbor keypoint)
	int new_keypoint_id = number_of_model_keypoints;
	for (int current_id = 0; current_id < this->current_keypoints.size(); current_id++)
	{
		// Check keypoint position
		if (this->current_keypoint_position[current_id].z < FLT_EPSILON)
		{
			this->current_match_to_model_id[current_id] = -1;
			continue;
		}

		//
		cv::Mat current_orb_feature = this->current_features.row(current_id);
		int min_feature_id = -1;
		int min_feature_distance = 64;	/* TODO init threshold */
		// Check each neighbor keypoint in model
		for (int model_id = 0; model_id < this->visible_point_model_index[current_id].size(); model_id++)
		{
			cv::Mat model_orb_feature = this->visible_model_features[current_id].row(model_id);
			int orb_distance = DescriptorDistance(model_orb_feature, current_orb_feature);
			if (min_feature_distance > orb_distance)
			{
				min_feature_distance = orb_distance;
				min_feature_id = model_id;
			}
		}

		if (this->current_match_to_model_id[current_id] >= 0)
		{
			/* Tracked by LK optic flow */
			// Alreadly matched, do nothing.
		}
		else if (min_feature_id >= 0)
		{
			/* Matched by feature */
			this->current_match_to_model_id[current_id] = visible_point_model_index[current_id][min_feature_id];
		}
		else if (this->visible_point_model_index[current_id].size() == 0)
		{
			/* New keypoint */
			this->current_match_to_model_id[current_id] = new_keypoint_id;
			new_keypoint_id++;
			//printf("2-  %d\n", current_id);
		}
		else
		{
			/* Invalid keypoint */
			this->current_match_to_model_id[current_id] = -1;
			//printf("3-  %d, %d\n", current_id, this->visible_point_model_index[current_id].size());
		}
	}




	// Reject repetitive matches (use LK optic flow result first)
	//std::vector<int> tracked_model_id;
	//for (int tracked_id = 0; tracked_id < this->number_of_tracked_keypoints; tracked_id++)
	//	if (this->current_match_to_model_id[tracked_id] >= 0)
	//	{	tracked_model_id.push_back(this->current_match_to_model_id[tracked_id]);	}
	//std::sort(tracked_model_id.begin(), tracked_model_id.end());
	//for (int new_detected_id = this->number_of_tracked_keypoints; new_detected_id < this->current_match_to_model_id.size(); new_detected_id++)
	//{
	//	if (this->current_match_to_model_id[new_detected_id] != -1)
	//	{
	//		bool is_find = std::binary_search(tracked_model_id.data(), tracked_model_id.data() + tracked_model_id.size(),
	//										  this->current_match_to_model_id[new_detected_id]);
	//		if (is_find)	this->current_match_to_model_id[new_detected_id] = -1;
	//	}
	//}
}





//
std::vector<float> compute_response(const cv::Mat& img, std::vector<cv::Point2f>& pts, int blockSize)
{
	std::vector<float> response_vector;
	response_vector.reserve(pts.size());

	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);
	//
	const uchar* ptr00 = img.ptr<uchar>();
	int r = blockSize / 2;
	int step = (int)(img.step / img.elemSize1());
	//
	cv::AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int* ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);
	//
	const float harris_k = 0.04f;
	float scale = 1.f / ((1 << 2) * blockSize * 255.f);
	float scale_sq_sq = scale * scale * scale * scale;


	for (int ptidx = 0; ptidx < pts.size(); ptidx++)
	{
		int x0 = cvRound(pts[ptidx].x);
		int y0 = cvRound(pts[ptidx].y);

		bool is_valid_point = true;
		if (x0 <= r || x0 >= (img.cols - r - 1))	is_valid_point = false;
		if (y0 <= r || y0 >= (img.rows - r - 1))	is_valid_point = false;

		if (is_valid_point)
		{
			const uchar* ptr0 = ptr00 + (y0 - r)*step + x0 - r;
			int a = 0, b = 0, c = 0;

			for (int k = 0; k < blockSize*blockSize; k++)
			{

				const uchar* ptr = ptr0 + ofs[k];
				int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
				int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
				a += Ix*Ix;
				b += Iy*Iy;
				c += Ix*Iy;
			}
			response_vector.push_back(((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq);
			//printf("%d : %f\r\n", ptidx, pts[ptidx].response);
			//min_response = min(min_response, pts[ptidx].response);
			//max_response = max(max_response, pts[ptidx].response);
		}
		else
		{
			response_vector.push_back(0);
		}
		//printf("%f , %f\r\n", min_response, max_response);
	}

	return response_vector;
}