#include "Data_loader.h"

//
#include <fstream>
#include "OurLib/My_quaternions_interface.h"


//
Data_loader::Data_loader()
{
	//printf("call Data_loader %d\n", this->ground_truth_camera_pose_index);
}
Data_loader::~Data_loader()
{
	delete this->image_loader;
}



// 
void Data_loader::init_image_loader(Image_loader * image_loader_ptr)
{
	if (image_loader_ptr == nullptr)
	{
		this->image_loader = new Blank_image_loader();
	}
	else
	{
		this->image_loader = image_loader_ptr;
	}
}


//
void Data_loader::load_ground_truth(string ground_truth_path, bool is_ICL_NUIM_data)
{
	ifstream ifile;

	ifile.open(ground_truth_path);
	if (ifile.is_open())
	{
		//
		Trajectory_node temp_GT;
		//
		string current_read_line;

		//
		this->with_ground_truth_trajectory = true;

		while (getline(ifile, current_read_line))
		{
			if (current_read_line.c_str()[0] == '#')	continue;
			//# index tx ty tz qw qx qy qz 
			//sscanf(current_read_line.c_str(), "%lf%f%f%f%f%f%f%f", 
			//	   &(temp_GT.time), &(temp_GT.tx), &(temp_GT.ty), &(temp_GT.tz), 
			//	   &(temp_GT.real), &(temp_GT.i), &(temp_GT.j), &(temp_GT.k));
			//# index tx ty tz qx qy qz qw 
			sscanf(current_read_line.c_str(), "%lf%f%f%f%f%f%f%f", 
				   &(temp_GT.time), &(temp_GT.tx), &(temp_GT.ty), &(temp_GT.tz), 
				   &(temp_GT.quaternions.qx), &(temp_GT.quaternions.qy), &(temp_GT.quaternions.qz), &(temp_GT.quaternions.qr));

			//printf("%lf, %f, %f, %f, %f, %f, %f, %f\r\n", 
			//	   temp_GT.time, (temp_GT.tx), (temp_GT.ty), (temp_GT.tz), 
			//	   (temp_GT.quaternions.qx), (temp_GT.quaternions.qy), (temp_GT.quaternions.qz), (temp_GT.quaternions.qr));

			// 
			this->ground_truth_trajectory.push_back(temp_GT);
		}
		ifile.close();
	}
	else
	{
		this->with_ground_truth_trajectory = false;
		printf("Can not open ground truth file!\r\n\r\n");
	}



	// is_ICL_NUIM_data : Transformate from left-hand coordinate to right-hand coordinate
	if (this->ground_truth_trajectory.size() > 0)
	{
		Trajectory_node first_node = this->ground_truth_trajectory[0];
		Eigen::Matrix3f first_rot;
		my_convert(first_rot, this->ground_truth_trajectory[0].quaternions);

		for (size_t node_id = 0; node_id < this->ground_truth_trajectory.size(); node_id++)
		{
			// ------------ Translation ------------
			// 
			this->ground_truth_trajectory[node_id].tx -= first_node.tx;
			this->ground_truth_trajectory[node_id].ty -= first_node.ty;
			this->ground_truth_trajectory[node_id].tz -= first_node.tz;
			// 
			if (is_ICL_NUIM_data)
			{
				this->ground_truth_trajectory[node_id].tx = +this->ground_truth_trajectory[node_id].tx;
				this->ground_truth_trajectory[node_id].ty = -this->ground_truth_trajectory[node_id].ty;
				this->ground_truth_trajectory[node_id].tz = +this->ground_truth_trajectory[node_id].tz;
			}
			

			// ------------ Rotation ------------
			//
			Eigen::Matrix3f rot_mat;
			my_convert(rot_mat, this->ground_truth_trajectory[node_id].quaternions);
			rot_mat = rot_mat.eval() * first_rot.inverse();
			// Coordinate transformation
			if (is_ICL_NUIM_data)
			{
				Eigen::Matrix3f coordinate_change;
				coordinate_change.setIdentity();
				coordinate_change(1, 1) = -1;
				rot_mat = coordinate_change * rot_mat.eval() * coordinate_change.inverse();
			}
			//
			my_convert(this->ground_truth_trajectory[node_id].quaternions, rot_mat);
		}
	}

}


//
bool Data_loader::load_next_frame(double & timestamp, cv::Mat & color_mat, cv::Mat & depth_mat, bool show_in_opencv)
{
	// Validate this->image_loader already initialized
	if (this->image_loader == nullptr)
	{
		printf("Data_engine::Data_loader::Image_loader has NOT been initalized!");
		return false;
	}

	// Load one frame
	bool load_state = this->image_loader->load_next_frame(timestamp, color_mat, depth_mat);
	
	// Visualization
	if (show_in_opencv)
	{
		this->show_current_frame(color_mat, depth_mat);
	}

	return load_state;
}


//
void Data_loader::show_current_frame(cv::Mat & color_mat, cv::Mat & depth_mat)
{
	cv::imshow("Color image", color_mat);
}


//
bool Data_loader::get_next_ground_truth_camera_pose(Trajectory_node & trajectory_node)
{
	//printf(" --------- %d------- %d\n", this->ground_truth_camera_pose_index, this->ground_truth_trajectory.size());
	
	if (this->ground_truth_camera_pose_index < this->ground_truth_trajectory.size()
		&& this->ground_truth_camera_pose_index >= 0)
	{
		trajectory_node = this->ground_truth_trajectory[this->ground_truth_camera_pose_index];
		this->ground_truth_camera_pose_index++;

		return true;
	}
	else
	{
		return false;
	}

}





