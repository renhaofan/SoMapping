/**
 *  @file Image_loader.h
 *  @brief Gather all frames pathes and concerned frame parametes, camera
 * intrinsic and extrinsic.
 *  @details Frame parametes(size, num_channel, elemSize, etc), camera intrinsic
 * and depth2color, depth2imu extrinsic. If you want to load frame, call
 * function l  bool load_next_frame(double &timestamp, cv::Mat &color_mat,
 * cv::Mat &depth_mat)
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include <cstdio>
#include <iostream>
using namespace std;
#include <cv.h>
#include <highgui.h>

#include <opencv2/opencv.hpp>

/**
 * @brief Base class for Blank_image_loader and Offline_image_loader.
 */
class Image_loader {
 public:
  /**
   * @brief The ImageLoaderMode enum, record which kind of file can be loaded.
   */
  enum ImageLoaderMode {
    /** No data found */
    NO_DATA,
    /** Only depth images found */
    WITH_DEPTH_ONLY,
    /** Both color and depth images found */
    WITH_COLOR_AND_DEPTH,
    /** Unequal number of color and depth */
    UNEQUAL_COLOR_AND_DEPTH_FRAMES,
    /** Number of enum */
    ILM_NUM
  };
  /** @brief Record chosen one of four image loader mode opetions. */
  ImageLoaderMode image_loader_mode;

  /**
   * @brief The ImageLoaderState enum, concerned with image loader state.
   */
  enum ImageLoaderState {
    /** Reach to the end-of-file */
    END_OF_DATA,
    /** Wait for loading the data */
    WAIT_FOR_DATA,
    /** Prepared to load the data */
    PREPARED_TO_LOAD,
    /** Number of enum */
    ILS_NUM
  };
  /** @brief Record image loader state, one of three states. */
  ImageLoaderState image_loader_state;

  /** @brief Default virtual constructor. */
  virtual ~Image_loader(){};

  /**
   * @brief Whether ready to load one frame.
   * @return true/false, success/failure.
   */
  virtual bool is_ready_to_load_next_frame() const = 0;
  /**
   * @brief Load one frame image(color and depth).
   * @param timestamp Timestamp of captured frame.
   * @param color_mat Color image matrix. (cv::Mat)
   * @param depth_mat Depth image matrix. (cv::Mat)
   * @return true/false, success/failure.
   */
  virtual bool load_next_frame(double &timestamp, cv::Mat &color_mat,
                               cv::Mat &depth_mat) = 0;

  /** @brief Extract depth image size by reference width and height. */
  virtual void get_depth_image_size(int &width, int &height) const = 0;
  /** @brief Extract color image size by reference width and height. */
  virtual void get_color_image_size(int &width, int &height) const = 0;
};

/**
 * @brief Class that is inherited using public inheritance. In case that
 * Image_loader pointer is nullptr.
 */
class Blank_image_loader : public Image_loader {
 public:
  /** @brief Default constructor.
   *  set this->image_loader_mode(NO_DATA).
   *  set this->image_loader_state(END_OF_DATA).
   */
  Blank_image_loader();
  /** @brief Default destructor. */
  ~Blank_image_loader();

  /**
   * @brief Override function, whether be ready to load next frame.
   * @return true/false, success/failure. false by default.
   */
  bool is_ready_to_load_next_frame() const override { return false; }
  /**
   * @brief Override function, load one frame image(color and depth).
   * @param timestamp Timestamp of captured frame.
   * @param color_mat Color image matrix. (cv::Mat)
   * @param depth_mat Depth image matrix. (cv::Mat)
   * @return true/false, success/failure. false by default.
   */
  bool load_next_frame(double &timestamp, cv::Mat &color_mat,
                       cv::Mat &depth_mat) override {
    return false;
  }

  /** @brief Extract depth image size by width and height which are set zero. */
  void get_depth_image_size(int &width, int &height) const override {
    width = 0;
    height = 0;
  }
  /** @brief Extract color image size by width and height which are set zero. */
  void get_color_image_size(int &width, int &height) const override {
    width = 0;
    height = 0;
  }
};

/**
 * @brief Class that is inherited using public inheritance. This class is used
 * normally. Gather all frames pathes and
 concerned frame parametes(size, num_channel, elemSize, etc), camera intrinsic
 and depth2color, depth2imu extrinsic. If you want to load frame, call function
 `load_next_frame(double &timestamp, cv::Mat &color_mat, cv::Mat &depth_mat)`.
 */
class Offline_image_loader : public Image_loader {
 public:
  /**
   * @brief The DatasetMode enum, control which kind of dataset.
   */
  enum DatasetMode {
    ICL = 0,               /** ICL-NUIM RGB-D Benchmark Dataset */
    TUM = 1,               /** TUM RGB-D SLAM Dataset and Benchmark */
    MyZR300 = 2,           /** MyZR300 Dataset */
    MyD435i = 3,           /** MyD435i Dataset */
    MyAzureKinect = 4,     /** MyAzureKinect Dataset */
    SCANNET = 5,           /** ScanNet Dataset */
    DATASETMODE_NUMBER = 6 /** The number of enum DatasetMode options. */
  };

  /** @brief Frame index. zero by default. */
  size_t frame_index = 0;
  /** @brief The number of frames. */
  size_t number_of_frames;

  /** @brief Color image width. */
  int color_width;
  /** @brief Color image height. */
  int color_height;
  /** @brief Color image per element(pixel in all channels) data size. */
  int color_element_size;
  /** @brief Color image number of channel */
  int color_channel_num;
  /** @brief Container with all color images' pathes. */
  vector<string> color_path_vector;
  /** @brief Container with all color images' timestamps. */
  vector<double> color_timestamp_vector;
  /** @brief Container with color image index.*/
  vector<int> color_index_vector;

  /** @brief Depth image width. */
  int depth_width;
  /** @brief Depth image height. */
  int depth_height;
  /** @brief Depth image per element(pixel in all channels) data size. */
  int depth_element_size;
  /** @brief Depth image number of channel */
  int depth_channel_num;
  /** @brief Container with all depth images' pathes. */
  vector<string> depth_path_vector;
  /** @brief Container with all depth images' timestamps. */
  vector<double> depth_timestamp_vector;
  /** @brief Container with depth image index.*/
  vector<int> depth_index_vector;

  /** @brief Container with images timestamp. */
  vector<double> image_timestamp_vector;

  /** @brief Default constructor. Do nothing.*/
  Offline_image_loader();
  /**
   * @brief Constructor with calibration file, dataset dir, datamode option.
   * @param cal Name of calibration file.
   * @param dir Dir path where dataset located in.
   * @param dm Datamode option.
   */
  Offline_image_loader(string cal, string dir, int dm);
  /** @brief Default destructor. Do nothing. */
  ~Offline_image_loader();

  /**
   * @brief Initialize frame parameters(size, channel, elemSize, etc) and
   * pathes.
   * @param color_folder Color images folder path.
   * @param depth_folder Depth images folder path.
   */
  void init(string color_folder, string depth_folder);
  /**
   * @brief Read frame width, height, elemSize, channle_num from dataset.
   */
  void read_image_parameters();
  /**
   * @brief Read frame intrinsic, frame size, depth2color dpeht2imu extrinsic.
   * @param cal File path where the calibration file is located.
   */
  void read_calibration_parameters(string cal);
  /**
   * @brief Get the frames file path, timestamp, the number of frames.
   *        And set ImageLoaderMode as WITH_COLOR_AND_DEPTH if no exception.
   * @param color_folder File path where the color sequences are located.
   * @param depth_folder File path where the depth sequences are located.
   */
  void detect_images(string color_folder, string depth_folder);
  /**
   * @brief Get the frames file path, timestamp, the number of frames.
   *        And set ImageLoaderMode as WITH_COLOR_AND_DEPTH if no exception.
   * @param associate associate.txt path, which records the correpondence
   *        between color and depth image.
   * @param colordir File path where the color sequences are located.
   * @param depthdir File path where the depth sequences are located.
   * @param dm enum DatasetMode, valid value range from 0 to 4.
   * @exception The number of depth images are not equal color images.
   */
  void detect_images(string associate, string colordir, string depthdir,
                     int dm);
  /**
   * @brief Print ImageLoaderMode and all frame path for DEBUG.
   * @param print_all_pathes Control whether print state, false by default.
   * @exception Incorrect ImageLoaderMode.
   */
  void print_state(bool print_all_pathes = false) const;
  /**
   * @brief Assignemnt this->frame_index.
   * @param frame_id Destination need to jump.
   * @exception (frame_id < 0) or (frame_id >= number_of_frames)
   * @return true/false, success/failure.
   */
  bool jump_to_specific_frame(int frame_id);

  /**
   * @brief Override function, whether be ready to load next frame.
   * @exception frame_Index >= number_of_frames.
   * @return true/false, success/failure.
   */
  bool is_ready_to_load_next_frame() const override;
  /**
   * @brief Override function, load one frame.
   * @param timestamp Timestamp of captured frame.
   * @param color_mat Color image matrix. (cv::Mat)
   * @param depth_mat Depth image matrix. (cv::Mat)
   * @exception frame_Index >= number_of_frames.
   * @exception Invalid ImageLoaderMode.
   * @return true/false, success/failure.
   * @warning If successful, color_mat will be converted from BGRA to BGR.
   */
  bool load_next_frame(double &timestamp, cv::Mat &color_mat,
                       cv::Mat &depth_mat) override;
  /**
   * @brief Extract depth width and height by two variables.
   * @param width Extract depth width.
   * @param height Extract depth height.
   */
  void get_depth_image_size(int &width, int &height) const override {
    width = this->depth_width;
    height = this->depth_height;
  }
  /**
   * @brief Extract color width and height by two variables.
   * @param width Extract color width.
   * @param height Extract color height.
   */
  void get_color_image_size(int &width, int &height) const override {
    width = this->color_width;
    height = this->color_height;
  }
};
