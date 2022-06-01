#include "Image_loader.h"

#include <SLAM_system/SLAM_system_settings.h>

#include "Log.h"

// file operation
#include <dirent.h>
#ifdef _WIN32
#include <io.h>
#elif __linux__
#include <inttypes.h>
#include <unistd.h>
#define __int64 int64_t
#define _close close
#define _read read
#define _lseek64 lseek64
#define _O_RDONLY O_RDONLY
#define _open open
#define _lseeki64 lseek64
#define _lseek lseek
#define stricmp strcasecmp
#endif
#include <fstream>
#include <string>
#include <vector>

// utils function
std::string append_slash_to_dirname(std::string dirname) {
  if (dirname.empty()) {
#ifdef LOGGING
    LOG_FATAL("Empty dirname: " + dirname);
    Log::shutdown();
#endif
    fprintf(stderr, "File %s, Line %d, Function %s(): Empty dirname.\n",
            __FILE__, __LINE__, __FUNCTION__);
    throw "Empty dirname!";
  }
  if (dirname[dirname.length() - 1] == '/') {
    return dirname;
  }
  return dirname + "/";
}

Blank_image_loader::Blank_image_loader() {
  this->image_loader_mode = ImageLoaderMode::NO_DATA;
  this->image_loader_state = ImageLoaderState::END_OF_DATA;
}
Blank_image_loader::~Blank_image_loader() {}

Offline_image_loader::Offline_image_loader() {}
Offline_image_loader::Offline_image_loader(const string cal, const string dir,
                                           int dm) {
  //  const char *imu = nullptr;
  string color_dir;
  string depth_dir;
  string associate_dir = dir + "/TIMESTAMP.txt";

#ifdef LOGGING
  LOG_INFO("<------ Initialising offline image loader ...");
#endif

  switch (dm) {
    case DatasetMode::ICL:
      //      color_dir = dir + "/rgb/";
      //      depth_dir = dir + "/depth/";
      color_dir = append_slash_to_dirname(dir) + "rgb/";
      depth_dir = append_slash_to_dirname(dir) + "depth/";
      detect_images(color_dir, depth_dir);
      this->read_calibration_parameters(cal);
      this->print_state(false);
      break;
    case DatasetMode::TUM:
      //      color_dir = dir + "/";
      //      depth_dir = dir + "/";
      color_dir = append_slash_to_dirname(dir);
      depth_dir = append_slash_to_dirname(dir);
      //      associate_dir = dir + "/associate.txt";
      associate_dir = append_slash_to_dirname(dir) + "associate.txt";
      detect_images(associate_dir, color_dir, depth_dir, dm);
      this->read_calibration_parameters(cal);
      this->print_state(false);
      break;
    case DatasetMode::MyZR300:
      //      color_dir = dir + "/color/";
      //      depth_dir = dir + "/filtered/";
      color_dir = append_slash_to_dirname(dir) + "color/";
      depth_dir = append_slash_to_dirname(dir) + "filtered/";
      detect_images(associate_dir, color_dir, depth_dir, dm);
      read_calibration_parameters(cal);
      this->print_state(false);
      break;
    case DatasetMode::MyD435i:
//      color_dir = dir + "/color/";
//      depth_dir = dir + "/depth/";
      color_dir = append_slash_to_dirname(dir) + "color/";
      depth_dir = append_slash_to_dirname(dir) + "depth/";
      detect_images(associate_dir, color_dir, depth_dir, dm);
      this->read_calibration_parameters(cal);
      this->print_state(false);
      break;
    case DatasetMode::MyAzureKinect:
//      color_dir = dir + "/color/";
//      depth_dir = dir + "/depth/";
      color_dir = append_slash_to_dirname(dir) + "color/";
      depth_dir = append_slash_to_dirname(dir) + "depth/";
      detect_images(associate_dir, color_dir, depth_dir, dm);
      this->read_calibration_parameters(cal);
      this->print_state(false);
      break;
    case DatasetMode::DATASETMODE_NUMBER:
    default:
#ifdef LOGGING
      LOG_FATAL("Undefined dataset!");
      Log::shutdown();
#endif
      fprintf(stderr, "File %s, Line %d, Function %s(), Undefined dataset\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Undefined dataset";
  }
}
Offline_image_loader::~Offline_image_loader() {}

void Offline_image_loader::init(string color_folder, string depth_folder) {
  // Detect images under folder
  this->detect_images(color_folder, depth_folder);

  // Read image parameters
  this->read_image_parameters();

  // Print loader state
  this->print_state(false);
}

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
  }
}

void Offline_image_loader::read_calibration_parameters(string cal) {
  if (cal.empty() || cal.length() == 0) {
#ifdef LOGGING
    LOG_WARNING(
        "Calibration filename not specified. Using default parameters.");
#endif
    printf("Calibration filename not specified. Using default parameters.\n");
    return;
  }
  std::ifstream f(cal);

  // Color camera parameters
  double fx, fy, cx, cy;
  f >> this->color_width >> this->color_height;
  f >> fx >> fy;
  f >> cx >> cy;
  SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy);

  // Depth camera parameters
  f >> this->depth_width >> this->depth_height;
  f >> fx >> fy;
  f >> cx >> cy;

  // depth2color, depth2imu
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

  // Read scale.
  string word;
  double scale;
  f >> word >> scale;
  SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy, 1.0f / scale);
  SLAM_system_settings::instance()->set_extrinsic(d2c, d2i);
}

void Offline_image_loader::detect_images(string color_folder,
                                         string depth_folder) {
  this->number_of_frames = 0;
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(color_folder.c_str())) != NULL) {
    while ((dirp = readdir(dp)) != NULL) {
      std::string name = std::string(dirp->d_name);

      if (name != "." && name != "..") this->number_of_frames++;
    }
    closedir(dp);
  }

  color_folder = append_slash_to_dirname(color_folder);
  depth_folder = append_slash_to_dirname(depth_folder);
  if (this->number_of_frames <= 1) {
    fprintf(stderr,
            "File %s, Line %d, Function %s(), the number of frames <= 1\n",
            __FILE__, __LINE__, __FUNCTION__);
    throw "the number of frames <= 1!";
  }
  for (size_t count = 0; count < this->number_of_frames - 1; count++) {
    this->color_path_vector.push_back(color_folder + to_string(count) + ".png");
    this->depth_path_vector.push_back(depth_folder + to_string(count) + ".png");
    this->image_timestamp_vector.push_back(
        (double(1.0 / 30.0) * double(count)));
  }

  // check mode of data loader
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
#ifdef LOGGING
      LOG_FATAL("Error occured when read dataset.");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(): Error occured when read "
              "dataset.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Error occured when read dataset!";
    }
  }
}

void Offline_image_loader::detect_images(string associate, string colordir,
                                         string depthdir, int dm) {
  ifstream inf;
  inf.open(associate, ifstream::in);
  if (!inf.is_open()) {
#ifdef LOGGING
    LOG_FATAL("Failed to open associate file: " + associate);
    Log::shutdown();
#endif
    fprintf(stderr, "File %s, Line %d, Function %s(): Failed to open file %s\n",
            __FILE__, __LINE__, __FUNCTION__, associate.c_str());
    throw "Failed to open file";
  }
  string line;
  size_t comma = 0;
  size_t comma2 = 0;

  switch (dm) {
    case DatasetMode::TUM:
      while (!inf.eof()) {
        getline(inf, line);

        comma = line.find(' ', 0);
        double timestamp = (double)atof(line.substr(0, comma).c_str());
        if (timestamp < 1e-3) continue;

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
      }

      // check mode of data loader
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
          this->image_loader_mode =
              ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES;
          this->number_of_frames = 0;
#ifdef LOGGING
          LOG_FATAL("Error occured when read dataset.");
          Log::shutdown();
#endif
          fprintf(stderr,
                  "File %s, Line %d, Function %s(): Error occured when read "
                  "dataset.\n",
                  __FILE__, __LINE__, __FUNCTION__);
          throw "Error occured when read dataset!";
        }
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
        if (timestamp < 1e-3) continue;

        this->image_timestamp_vector.push_back(timestamp * 1e-6);
        comma2 = line.find('g', comma + 1);
        string imgName = line.substr(comma + 1, comma2 - comma).c_str();
        this->color_path_vector.push_back(colordir + imgName);
        this->depth_path_vector.push_back(depthdir + imgName);
        this->image_loader_mode = ImageLoaderMode::WITH_COLOR_AND_DEPTH;
      }
      // check mode of data loader
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
          this->image_loader_mode =
              ImageLoaderMode::UNEQUAL_COLOR_AND_DEPTH_FRAMES;
          this->number_of_frames = 0;
#ifdef LOGGING
          LOG_FATAL("Error occured when read dataset.");
          Log::shutdown();
#endif
          fprintf(stderr,
                  "File %s, Line %d, Function %s(): Error occured when read "
                  "dataset.\n",
                  __FILE__, __LINE__, __FUNCTION__);
          throw "Error occured when read dataset!";
        }
      }

      break;
    case DATASETMODE_NUMBER:
    default:
#ifdef LOGGING
      LOG_FATAL("Invalid dataset option.");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(): Invalid dataset "
              "option. Valid option from 0 to %d. \n",
              __FILE__, __LINE__, __FUNCTION__,
              DatasetMode::DATASETMODE_NUMBER - 1);
      throw "Invalid dataset option.";
      break;
  }

  inf.close();
}

void Offline_image_loader::print_state(bool print_all_pathes) const {
  switch (this->image_loader_mode) {
    case ImageLoaderMode::NO_DATA: {
#ifdef LOGGING
      LOG_FATAL("ImageLoaderMode: NO_DATA");
      Log::shutdown();
#endif
      fprintf(
          stderr,
          "File %s, Line %d, Function %s(), No color or depth image found!\n",
          __FILE__, __LINE__, __FUNCTION__);
      throw "No color or depth image found!";
      break;
    }
    case ImageLoaderMode::WITH_DEPTH_ONLY: {
#ifdef LOGGING
      LOG_FATAL("ImageLoaderMode: WITH_DEPTH_ONLY");
      Log::shutdown();
#endif
      // print all depth image pathes
      if (print_all_pathes) {
        for (size_t path_id = 0; path_id < this->depth_path_vector.size();
             path_id++) {
          printf("%s\n", this->depth_path_vector[path_id].c_str());
        }
      }

      fprintf(stderr,
              "File %s, Line %d, Function %s(), Only %zu depth images found!\n",
              __FILE__, __LINE__, __FUNCTION__, this->depth_path_vector.size());
      throw "Only depth images found!";
      break;
    }
    case ImageLoaderMode::WITH_COLOR_AND_DEPTH: {
#ifdef LOGGING
      LOG_INFO("ImageLoaderMode: WITH_COLOR_AND_DEPTH");
      LOG_INFO("Depth images number: " +
               to_string(this->depth_path_vector.size()));
      LOG_INFO("Color images number: " +
               to_string(this->color_path_vector.size()));
      LOG_INFO("Initialising offline image loader finished ------>");
#endif
      // print all frames pathes
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
#ifdef LOGGING
      LOG_INFO("ImageLoaderMode: UNEQUAL_COLOR_AND_DEPTH_FRAMES");
      LOG_INFO("Depth images " + to_string(this->depth_path_vector.size()));
      LOG_FATAL("Color images " + to_string(this->color_path_vector.size()));
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(), Unequal color and depth images "
              "found!\n (%zu depth images != %zu color images)\n",
              __FILE__, __LINE__, __FUNCTION__, this->depth_path_vector.size(),
              this->color_path_vector.size());
      throw "Unequal color and depth images!";
      break;
    }

    case ImageLoaderMode::ILM_NUM:
    default: {
#ifdef LOGGING
      LOG_FATAL("Invalid ImageLoaderMode option!");
      Log::shutdown();
#endif

      fprintf(stderr,
              "File %s, Line %d, Function %s(), Invalid ImageLoaderMode.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw "Invalid ImageLoaderMode!";
      break;
    }
  }
}

bool Offline_image_loader::jump_to_specific_frame(int frame_id) {
  if ((frame_id < 0) ||
      (static_cast<size_t>(frame_id) >= this->number_of_frames)) {
#ifdef LOGGING
    LOG_ERROR("Invalid frame id: " + std::to_string(frame_id));
#endif
    fprintf(stderr, "File %s, Line %d, Function %s(), Invalid frame id",
            __FILE__, __LINE__, __FUNCTION__);
    return false;
  }
  this->frame_index = static_cast<size_t>(frame_id);
  return true;
}

bool Offline_image_loader::is_ready_to_load_next_frame() const {
  if (this->frame_index > this->number_of_frames) {
#ifdef LOGGING
    LOG_ERROR("Invalid frame id: " + std::to_string(this->frame_index));
#endif
    fprintf(stderr, "File %s, Line %d, Function %s(), Invalid frame index.\n",
            __FILE__, __LINE__, __FUNCTION__);
    throw "Invalid frame index!";
  } else if (this->frame_index == this->number_of_frames) {
    // More likely, run to the flag (End of process) rather than BUG.
    return false;
  }
  return true;
}

bool Offline_image_loader::load_next_frame(double &timestamp,
                                           cv::Mat &color_mat,
                                           cv::Mat &depth_mat) {
  // Validate frame index
  if (!this->is_ready_to_load_next_frame()) return false;

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
