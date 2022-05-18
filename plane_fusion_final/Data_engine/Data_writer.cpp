#include "Data_writer.h"



//!
Data_writer::Data_writer()
{
}

Data_writer::~Data_writer()
{
}


//! Init
void Data_writer::init(string output_folder, bool _is_ICL_NUIM_dataset)
{
	//
	this->output_folder = output_folder;

	//
	this->is_ICL_NUIM_dataset = _is_ICL_NUIM_dataset;

}


//
void Data_writer::save_trajectory(const Trajectory & estimated_trajectory)
{
	// Open file
	FILE * file_pointer = fopen((this->output_folder + "/estimate_trajectory.txt\0").c_str(), "w+");
	// 
	if (file_pointer)
	{
		for (int frame_counter = 0; frame_counter < estimated_trajectory.size(); frame_counter++)
		{
			Trajectory_node node_to_save(estimated_trajectory[frame_counter]);

			//! Coordinate convert
			if (this->is_ICL_NUIM_dataset)
			{
				// ------------ Translation ------------
				node_to_save.tx = +node_to_save.tx;
				node_to_save.ty = -node_to_save.ty;
				node_to_save.tz = +node_to_save.tz;
				
				// ------------ Rotation ------------
				Eigen::Matrix3f rot_mat;
				my_convert(rot_mat, node_to_save.quaternions);
				// Coordinate transformation
				Eigen::Matrix3f coordinate_change;
				coordinate_change.setIdentity();
				coordinate_change(1, 1) = -1;
				rot_mat = coordinate_change * rot_mat.eval() * coordinate_change.inverse();
				//
				my_convert(node_to_save.quaternions, rot_mat);
			}

			// Save one trajectory node 
			fprintf(file_pointer, "%lf %f %f %f %f %f %f %f\r\n",
					node_to_save.time, 
					node_to_save.tx, 
					node_to_save.ty, 
					node_to_save.tz,
					node_to_save.quaternions.qx, 
					node_to_save.quaternions.qy,
					node_to_save.quaternions.qz, 
					node_to_save.quaternions.qr);
		}
		printf("Estimate trajectory saved.(estimate with %d frame)\r\n", estimated_trajectory.size());
		// Close file
		fclose(file_pointer);
	}
	else
	{
		printf("Can not open %s\r\n", (this->output_folder + "/estimate_trajectory.txt\0").c_str());
	}

}