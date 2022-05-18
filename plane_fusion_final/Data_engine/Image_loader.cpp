#include "Image_loader.h"

// file operation
#include <SLAM_system/SLAM_system_settings.h>
#include <dirent.h>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>

// ------------------------------------- Blank_image_loader
// -------------------------------------
//
Blank_image_loader::Blank_image_loader() {
  this->image_loader_mode = ImageLoaderMode::NO_DATA;
  this->image_loader_state = ImageLoaderState::END_OF_DATA;
}
Blank_image_loader::~Blank_image_loader() {}
std::string append_slash_to_dirname(std::string dirname) {
  if (dirname[dirname.length() - 1] == '/') {
    return dirname;
  }
  return dirname + "/";
}

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
Offline_image_loader::Offline_image_loader(const string cal, const string dir,
                                           int dm) {
  const char *imu = nullptr;
  string color_dir;
  string depth_dir;
  string associate_dir = dir + "/TIMESTAMP.txt";

  printf("init offline image loader ...\n");

  switch (dm) {
  case DatasetMode::ICL:
    color_dir = dir + "/rgb/";
    depth_dir = dir + "/depth/";
    detect_images(color_dir, depth_dir);
    // Read image parameters
    this->read_calibration_parameters(cal);
    // Print loader state
    this->print_state(false);

    break;
  case DatasetMode::TUM:
    color_dir = dir + "/";
    depth_dir = dir + "/";
    associate_dir = dir + "/associate.txt";
    detect_images(associate_dir, color_dir, depth_dir, dm);
    // Read image parameters
    this->read_calibration_parameters(cal);
    // Print loader state
    this->print_state(false);

    break;
  case DatasetMode::MyZR300:
    color_dir = dir + "/color/";
    depth_dir = dir + "/filtered/";
    detect_images(associate_dir, color_dir, depth_dir, dm);
    // Read image parameters
    read_calibration_parameters(cal);
    // Print loader state
    this->print_state(false);

    break;
  case DatasetMode::MyD435i:
    color_dir = dir + "/color/";
    depth_dir = dir + "/depth/";
    detect_images(associate_dir, color_dir, depth_dir, dm);
    // Read image parameters
    this->read_calibration_parameters(cal);
    // Print loader state
    this->print_state(false);

    break;
  case DatasetMode::MyAzureKinect:
    color_dir = dir + "/color/";
    depth_dir = dir + "/depth/";
    detect_images(associate_dir, color_dir, depth_dir, dm);
    // Read image parameters
    this->read_calibration_parameters(cal);
    // Print loader state
    this->print_state(false);

    break;
  default:
    cout << "unknown dataset" << endl;
  }
}
//
bool Offline_image_loader::load_next_frame(double &timestamp,
                                           cv::Mat &color_mat,
                                           cv::Mat &depth_mat) {
  // Validate frame index
  if (!this->is_ready_to_load_next_frame())
    return false;

  // Load timestamp and images
  timestamp = this->image_timestamp_vector[this->frame_index];
  switch (this->image_loader_mode) {
  case ImageLoaderMode::NO_DATA:
    return false;
  case ImageLoaderMode::WITH_DEPTH_ONLY: {
    depth_mat = cv::imread(this->depth_path_vector[this->frame_index].c_str(),
                           CV_LOAD_IMAGE_UNCHANGED);
    break;
  }
  case ImageLoaderMode::WITH_COLOR_AND_DEPTH: {
    depth_mat = cv::imread(this->depth_path_vector[this->frame_index].c_str(),
                           CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat temp =
        cv::imread(this->color_path_vector[this->frame_index].c_str(),
                   CV_LOAD_IMAGE_UNCHANGED);
    cv::cvtColor(temp, color_mat, cv::COLOR_BGRA2BGR);
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
// Offline_image_loader::Offline_image_loader(string color_folder, string
// depth_folder)
//{
//	//
//	this->init(color_folder, depth_folder);
//}

void Offline_image_loader::read_calibration_parameters(string cal) {
  if (cal.empty() || cal.length() == 0) {
    printf("Calibration filename not specified. Using default parameters.\n");
    return;
  }

  std::ifstream f(cal);
  // rgb
  double fx, fy, cx, cy;
  f >> this->color_width >> this->color_height;
  f >> fx >> fy;
  f >> cx >> cy;
  SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy);

  f >> this->depth_width >> this->depth_height;
  f >> fx >> fy;
  f >> cx >> cy;

  My_Type::Matrix44f d2c, d2i;

  f >> d2c.m00 >> d2c.m10 >> d2c.m20 >> d2c.m30;
  f >> d2c.m01 >> d2c.m11 >> d2c.m21 >> d2c.m31;
  f >> d2c.m02 >> d2c.m12 >> d2c.m22 >> d2c.m32;
  d2c.m03 = 0.0f;
  d2c.m13 = 0.0f;
  d2c.m23 = 0.0f;
  d2c.m33 = 1.0f;
  f >> d2i.m00 >> d2i.m10 >> d2i.m20 >> d2i.m30;
  f >> d2i.m01 >> d2i.m11 >> d2i.m21 >> d2i.m31;
  f >> d2i.m02 >> d2i.m12 >> d2i.m22 >> d2i.m32;
  d2i.m03 = 0.0f;
  d2i.m13 = 0.0f;
  d2i.m23 = 0.0f;
  d2i.m33 = 1.0f;
  string word;
  double scale;
  f >> word >> scale;
  SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy, 1.0f / scale);
  SLAM_system_settings::instance()->set_extrinsic(d2c, d2i);

  //    cout << SLAM_system_settings::instance()->color_params.sensor_fx <<
  //    endl; cout << SLAM_system_settings::instance()->depth_params.sensor_fx
  //    << endl; cout <<
  //    SLAM_system_settings::instance()->depth_params.sensor_scale << endl;
  //    cout << SLAM_system_settings::instance()->depth2color_mat << endl;
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
void Offline_image_loader::detect_images(string associate, string colordir,
                                         string depthdir, int dm) {
  ifstream inf;
  inf.open(associate, ifstream::in);

  string line;
  size_t comma = 0;
  size_t comma2 = 0;

  switch (dm) {
  case DatasetMode::TUM:
    while (!inf.eof()) {
      getline(inf, line);

      comma = line.find(' ', 0);
      double timestamp = (double)atof(line.substr(0, comma).c_str());
      if (timestamp < 1e-3)
        continue;

      comma2 = line.find(' ', comma + 1);
      string colorName = line.substr(comma + 1, comma2 - comma - 1).c_str();

      comma = line.find(' ', comma2 + 1);
      string temp1 = line.substr(comma2 + 1, comma - comma2 - 1);
      timestamp = (double)atof(temp1.c_str());

      comma2 = line.find('g', comma + 1);
      string depthName = line.substr(comma + 1, comma2 - comma).c_str();

      this->image_timestamp_vector.push_back(timestamp);
      this->color_path_vector.push_back(colordir + colorName);
      this->depth_path_vector.push_back(depthdir + depthName);
      this->image_loader_mode = ImageLoaderMode::WITH_COLOR_AND_DEPTH;
      this->number_of_frames = this->depth_path_vector.size();
    }

    break;
  case DatasetMode::MyZR300:
  case DatasetMode::MyD435i:
  case DatasetMode::MyAzureKinect:
    getline(inf, line);
    while (!inf.eof()) {
      getline(inf, line);

      comma = line.find(',', 0);
      string temp1 = line.substr(0, comma);
      double timestamp = (double)atof(temp1.c_str());
      if (timestamp < 1e-3)
        continue;

      this->image_timestamp_vector.push_back(timestamp * 1e-6);
      comma2 = line.find('g', comma + 1);
      string imgName = line.substr(comma + 1, comma2 - comma).c_str();
      this->color_path_vector.push_back(colordir + imgName);
      this->depth_path_vector.push_back(depthdir + imgName);
      this->image_loader_mode = ImageLoaderMode::WITH_COLOR_AND_DEPTH;
      this->number_of_frames = this->depth_path_vector.size();
    }
    break;
  default
      : // my dataset, i.e. the color and depth image have same timestamp(name).
    break;
  }

  inf.close();
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

//
void Offline_image_loader::detect_images(string color_folder,
                                         string depth_folder) {

  /*//Color folder
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(color_folder.c_str())) != NULL)
  {

      while ((dirp = readdir(dp)) != NULL) {
          std::string name = std::string(dirp->d_name);

          if (name != "." && name != "..")
              this->color_path_vector.push_back(name);
      }
      closedir(dp);


      std::sort(this->color_path_vector.begin(), this->color_path_vector.end());
      color_folder = append_slash_to_dirname(color_folder);
      for (unsigned int i = 0; i < this->color_path_vector.size(); i++) {
          if (this->color_path_vector[i].at(0) != '/')
              this->color_path_vector[i] = color_folder +
  this->color_path_vector[i];
      }
  }

  //Depth folder
  DIR *dp1;
  struct dirent *dirp1;
  if((dp1  = opendir(depth_folder.c_str())) != NULL)
  {
      while ((dirp1 = readdir(dp1)) != NULL) {
          std::string name = std::string(dirp1->d_name);
          if (name != "." && name != "..")
              this->depth_path_vector.push_back(name);
      }
      closedir(dp1);


      std::sort(this->depth_path_vector.begin(), this->depth_path_vector.end());
      depth_folder = append_slash_to_dirname(depth_folder);
      for (unsigned int i = 0; i < this->depth_path_vector.size(); i++) {

              // save time stamp
              int pos2 = this->depth_path_vector[i].find_last_of(".");

              double TUMTimeStamp =
  std::atof(this->depth_path_vector[i].substr(0,pos2-1).c_str());
              this->image_timestamp_vector.push_back(TUMTimeStamp);

              this->depth_path_vector[i] = depth_folder +
  this->depth_path_vector[i];


      }
  }*/
  this->number_of_frames = 0;
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(color_folder.c_str())) != NULL) {
    while ((dirp = readdir(dp)) != NULL) {
      std::string name = std::string(dirp->d_name);

      if (name != "." && name != "..")
        this->number_of_frames++;
    }
    closedir(dp);
  }

  color_folder = append_slash_to_dirname(color_folder);
  depth_folder = append_slash_to_dirname(depth_folder);
  for (int count = 0; count < this->number_of_frames - 1; count++) {
    this->color_path_vector.push_back(color_folder + to_string(count) + ".png");
    this->depth_path_vector.push_back(depth_folder + to_string(count) + ".png");
    this->image_timestamp_vector.push_back(
        (double(1.0 / 30.0) * double(count)));
  }

  // Set mode of data loader
  if (this->depth_path_vector.size() == 0 &&
      this->color_path_vector.size() == 0) {
    this->image_loader_mode = ImageLoaderMode::NO_DATA;
    this->number_of_frames = 0;
  } else if (this->depth_path_vector.size() > 0 &&
             this->color_path_vector.size() == 0) {
    this->image_loader_mode = ImageLoaderMode::WITH_DEPTH_ONLY;
    this->number_of_frames = this->depth_path_vector.size();
  } else if (this->depth_path_vector.size() > 0 &&
             this->color_path_vector.size() > 0) {
    // Validate number of loaded frames
    if (this->color_path_vector.size() == this->depth_path_vector.size()) {
      this->image_loader_mode = ImageLoaderMode::WITH_COLOR_AND_DEPTH;
      this->number_of_frames = this->depth_path_vector.size();
    } else {
      this->image_loader_mode = ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES;
      this->number_of_frames = 0;
    }
  }
}

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
