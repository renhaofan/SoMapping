#include "Image_loader.h"

#include <iostream>
// file operation
#include <fstream>
#include <io.h>
#include <stdio.h>
#include <string>
#include <vector>

#define _GLIBCXX_USE_CXX11_ABI 0
// ------------------------------------- Blank_image_loader
// -------------------------------------
//
Blank_image_loader::Blank_image_loader() {
  this->image_loader_mode = ImageLoaderMode::NO_DATA;
  this->image_loader_state = ImageLoaderState::END_OF_DATA;
}
Blank_image_loader::~Blank_image_loader() {}

// ------------------------------------- Offline_image_loader
// -------------------------------------

// --- Interfaces :

//
bool Offline_image_loader::is_ready_to_load_next_frame() const {
  // Validate frame index
  if (this->frame_index < 0 || this->frame_index >= this->number_of_frames)
    return false;
  else
    return true;
}

//
bool Offline_image_loader::load_next_frame(double &timestamp,
                                           cv::Mat &color_mat,
                                           cv::Mat &depth_mat) {
  // Validate frame index
  if (!this->is_ready_to_load_next_frame())
    return false;

  // Load timestamp and images
  timestamp = this->depth_timestamp_vector[this->frame_index];
  switch (this->image_loader_mode) {
  case ImageLoaderMode::NO_DATA:
    return false;
  case ImageLoaderMode::WITH_DEPTH_ONLY: {
    depth_mat = cv::imread(this->depth_path_vector[this->frame_index].c_str(),
                           CV_LOAD_IMAGE_UNCHANGED);
    break;
  }
  case ImageLoaderMode::WITH_COLOR_AND_DEPTH: {
    int depth_index = this->depth_index_vector[this->frame_index];
    int color_index = this->color_index_vector[this->frame_index];
    depth_mat = cv::imread(this->depth_path_vector[depth_index].c_str(),
                           CV_LOAD_IMAGE_UNCHANGED);
    color_mat = cv::imread(this->color_path_vector[color_index].c_str(),
                           CV_LOAD_IMAGE_UNCHANGED);
    break;
  }
  case ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES:
    return false;
  default:
    break;
  }

  // To next frame
  this->frame_index++;
  return true;
}

// --- Member functions :
//
Offline_image_loader::Offline_image_loader() {}
Offline_image_loader::~Offline_image_loader() {}
//
Offline_image_loader::Offline_image_loader(string color_folder,
                                           string depth_folder) {
  //
  this->init(color_folder, depth_folder);
}

//
void Offline_image_loader::init(string color_folder, string depth_folder) {
  // Detect images under folder
  this->detect_images(color_folder, depth_folder);

  // Read image parameters
  this->read_image_parameters();

  // Print loader state
  this->print_state(false);
}

//
bool Offline_image_loader::jump_to_specific_frame(int frame_id) {

  // Validate frame index
  if (frame_id < 0 || frame_id >= this->number_of_frames)
    return false;
  this->frame_index = frame_id;
  return true;

  //// Unnecessary
  // switch (this->image_loader_mode)
  //{
  //	case ImageLoaderMode::NO_DATA:
  //		return false;
  //	case ImageLoaderMode::WITH_DEPTH_ONLY:
  //	{
  //		depth_mat =
  //imread(this->depth_path_vector[this->frame_index].c_str(),
  //CV_LOAD_IMAGE_UNCHANGED); 		break;
  //	}
  //	case ImageLoaderMode::WITH_COLOR_AND_DEPTH:
  //	{
  //		depth_mat =
  //imread(this->depth_path_vector[this->frame_index].c_str(),
  //CV_LOAD_IMAGE_UNCHANGED); 		color_mat =
  //imread(this->color_path_vector[this->frame_index].c_str(),
  //CV_LOAD_IMAGE_UNCHANGED); 		break;
  //	}
  //	case ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES:
  //		return false;
  //	default:
  //		break;
  //}
}

#define _A_SUBDIR 0x10
//
// void Offline_image_loader::detect_images(string color_folder, string
// depth_folder)
//{
//
//	// File handle
//	long long hFile = 0;
//
//	// File information
//	struct _finddata_t fileinfo;
//	string path;
//
//	// Color folder
//	if (color_folder.length() > 0)
//	{
//		//
//		if ((hFile =
//_findfirst(path.assign(color_folder).append("/*").c_str(), &fileinfo)) != -1)
//		{
//			do
//			{
//				if ((fileinfo.attrib & _A_SUBDIR)){ /* space holder */
//} 				else
//				{
//					this->color_path_vector.push_back(color_folder +
//fileinfo.name);
//					// save time stamp
//					double time_stamp = 0.0;
//					sscanf(fileinfo.name, "%lf",
//&time_stamp); 					this->color_timestamp_vector.push_back(time_stamp);
//				}
//
//			} while (_findnext(hFile, &fileinfo) == 0);
//
//			_findclose(hFile);
//		}
//	}
//
//
//	// Depth folder
//	if (depth_folder.length() > 0)
//	{
//		//
//		if ((hFile =
//_findfirst(path.assign(depth_folder).append("/*").c_str(), &fileinfo)) != -1)
//		{
//			do
//			{
//				if ((fileinfo.attrib & _A_SUBDIR)){ /* space holder */
//} 				else
//				{
//					this->depth_path_vector.push_back(depth_folder +
//fileinfo.name);
//					// save time stamp
//					double time_stamp = 0.0;
//					sscanf(fileinfo.name, "%lf",
//&time_stamp); 					this->depth_timestamp_vector.push_back(time_stamp);
//				}
//
//			} while (_findnext(hFile, &fileinfo) == 0);
//
//			_findclose(hFile);
//		}
//	}
//
//
//	// Set mode of data loader
//	if (this->depth_path_vector.size() == 0 &&
//this->color_path_vector.size() == 0)
//	{
//		this->image_loader_mode = ImageLoaderMode::NO_DATA;
//		this->number_of_frames = 0;
//	}
//	else if (this->depth_path_vector.size() > 0 &&
//this->color_path_vector.size() == 0)
//	{
//		this->image_loader_mode = ImageLoaderMode::WITH_DEPTH_ONLY;
//		this->number_of_frames = this->depth_path_vector.size();
//	}
//	else if (this->depth_path_vector.size() > 0 &&
//this->color_path_vector.size() > 0)
//	{
//		this->image_loader_mode = ImageLoaderMode::WITH_COLOR_AND_DEPTH;
//		// algin timestamp (align to depth image)
//		this->depth_index_vector.resize(this->depth_path_vector.size());
//		for (int i = 0; i < this->depth_path_vector.size(); i++)
//			this->depth_index_vector[i] = i;
//		//
//		this->color_index_vector.reserve(this->depth_path_vector.size());
//		for (int depth_image_id = 0, color_image_id = 0;
//			 depth_image_id < this->depth_timestamp_vector.size() &&
//color_image_id < this->color_timestamp_vector.size(); 			 depth_image_id++)
//		{
//			double depth_timestamp =
//this->depth_timestamp_vector[depth_image_id];
//
//			while (color_image_id <
//this->color_timestamp_vector.size())
//			{
//				if (this->color_timestamp_vector[color_image_id] <
//depth_timestamp) 				{	color_image_id++;} 				else
//				{
//					this->color_index_vector.push_back(color_image_id);
//					break;
//				}
//			}
//		}
//		//
//		this->depth_index_vector.resize(this->color_index_vector.size());
//		this->number_of_frames = this->depth_index_vector.size();
//	}
//
//}

//
void Offline_image_loader::read_image_parameters() {

  if (this->color_path_vector.size() > 0) {
    cv::Mat temp_mat;

    // Read color images
    temp_mat =
        cv::imread(this->color_path_vector[0].c_str(), CV_LOAD_IMAGE_UNCHANGED);

    // Read parameters of color image
    this->color_width = temp_mat.cols;
    this->color_height = temp_mat.rows;
    this->color_element_size = (int)temp_mat.elemSize();
    this->color_channel_num = temp_mat.channels();

    //
    // cout << "color_width\t" << this->color_width << "\tcolor_height\t" <<
    // this->color_height << endl; cout << "color_element_size\t" <<
    // this->color_element_size << endl; cout << "color_channel_num\t" <<
    // this->color_channel_num << endl << endl;

    // Show color image in OpenCV window
    // imshow("this->temp_mat", this->temp_mat);
  }

  if (this->depth_path_vector.size() > 0) {
    cv::Mat temp_mat;

    // Read depth images
    temp_mat =
        cv::imread(this->depth_path_vector[0].c_str(), CV_LOAD_IMAGE_UNCHANGED);

    // Read parameters of depth image
    this->depth_width = temp_mat.cols;
    this->depth_height = temp_mat.rows;
    this->depth_element_size = (int)temp_mat.elemSize();
    this->depth_channel_num = temp_mat.channels();

    //
    // cout << "depth_width\t" << this->depth_width << "\tdepth_height\t" <<
    // this->depth_height << endl; cout << "depth_element_size\t" <<
    // this->depth_element_size << endl; cout << "depth_channel_num\t" <<
    // this->depth_channel_num << endl << endl;
  }
}

//
void Offline_image_loader::print_state(bool print_all_pathes) const {
  switch (this->image_loader_mode) {
  case ImageLoaderMode::NO_DATA: {
    printf("No color or depth image found!\n");
    break;
  }
  case ImageLoaderMode::WITH_DEPTH_ONLY: {
    printf("Only depth images found!\n %d depth images\n",
           (int)this->depth_path_vector.size());

    if (print_all_pathes) {
      for (size_t path_id = 0; path_id < this->depth_path_vector.size();
           path_id++) {
        printf("%s\n", this->depth_path_vector[path_id].c_str());
      }
    }
    break;
  }
  case ImageLoaderMode::WITH_COLOR_AND_DEPTH: {
    printf("Both color and depth images found!\n %d depth images \n %d color "
           "images\n",
           (int)this->depth_path_vector.size(),
           (int)this->color_path_vector.size());

    if (print_all_pathes) {
      printf("Depth image pathes:\n");
      for (size_t path_id = 0; path_id < this->depth_path_vector.size();
           path_id++) {
        printf("%s\n", this->depth_path_vector[path_id].c_str());
      }
      printf("\n");

      printf("Color image pathes:\n");
      for (size_t path_id = 0; path_id < this->color_path_vector.size();
           path_id++) {
        printf("%s\n", this->color_path_vector[path_id].c_str());
      }
      printf("\n");
    }
    break;
  }
  case ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES: {
    printf("Error data format : Unequal color and depth images found!\n (%d "
           "depth images != %d color images)\n",
           (int)this->depth_path_vector.size(),
           (int)this->color_path_vector.size());
    break;
  }
  default: {
    cout << "Invalid state!" << endl;
    break;
  }
  }
}

void Offline_image_loader::_findclose(long long int file) {}

void Offline_image_loader::detect_images(string color_folder,
                                         string depth_folder) {}
