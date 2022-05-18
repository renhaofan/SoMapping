#pragma once

// OpenCV
#include <cv.h>
#include <float.h>
#include <highgui.h>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "Preprocess_KernelFunc.cuh"
//
#include "Hierarchy_image.h"
#include "OurLib/My_matrix.h"

//
#include "SLAM_system/SLAM_system_settings.h"

//!
/*!

*/
class Preprocess_engine {
 public:
  //!
  My_Type::Vector2i raw_color_size;
  //!
  My_Type::Vector2i raw_depth_size;
  //!
  int image_alignment_width;
  //!
  My_Type::Vector2i aligned_depth_size;

#pragma region(HOST)

  // ------------------- Depth information
  //!
  My_Type::Vector3f *raw_aligned_points;
  //! Points generate from filtered points
  Hierarchy_image<My_Type::Vector3f> hierarchy_points;
  //! Normal vectors compute from filtered depth
  Hierarchy_image<My_Type::Vector3f> hierarchy_normals;
  //! Points generate from model
  Hierarchy_image<My_Type::Vector3f> hierarchy_model_points;
  //! Normal vectors compute from model
  Hierarchy_image<My_Type::Vector3f> hierarchy_model_normals;

  // ------------------- Intensity information
  //!
  Hierarchy_image<float> hierarchy_intensity;
  //!
  Hierarchy_image<My_Type::Vector2f> hierarchy_gradient;
  //!
  Hierarchy_image<float> hierarchy_model_intensity;
  //!
  Hierarchy_image<My_Type::Vector2f> hierarchy_model_gradient;

#pragma endregion

#pragma region(CUDA)

  // ------------------- Depth information
  //! Device pointer of Depth image (unsigned short)
  RawDepthType *dev_raw_depth;
  //! Device pointer of float-type depth image buffer
  float *dev_depth_buffer;
  //! Device pointer of Depth image (unsigned short)
  float *dev_filtered_depth;
  //! Aligned current points
  My_Type::Vector3f *dev_raw_aligned_points;
  //!
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_points;
  //!
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_normals;

  //! Points generate from model
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_model_points;
  //! Normal vectors compute from model
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_model_normals;

  // ------------------- Intensity information
  //! Device pointer of Color image (RGB)
  RawColorType *dev_raw_color;
  //!
  Hierarchy_image<float> dev_hierarchy_intensity;
  //!
  Hierarchy_image<My_Type::Vector2f> dev_hierarchy_gradient;
  //!
  Hierarchy_image<float> dev_hierarchy_model_intensity;
  //!
  Hierarchy_image<My_Type::Vector2f> dev_hierarchy_model_gradient;

#pragma endregion

  //!
  virtual void generate_render_information() = 0;

  //! Default constructor/destructor
  virtual ~Preprocess_engine(){};

  //! Initialization
  /*!
          \param	raw_color_size			Size of raw color image

          \param	raw_depth_size			Size of raw depth image

          \param	image_alignment_width	Image alignment unit.(PIXEL)

          \param	number_of_layers		Number of hierarchy
     image layers.

          \return	void
  */
  virtual void init(My_Type::Vector2i raw_color_size,
                    My_Type::Vector2i raw_depth_size, int image_alignment_width,
                    int number_of_layers);

  //! Preprocess image
  /*!
          \param	raw_color	Raw color image captured by sensor.

          \param	raw_depth	Raw depth image captured by sensor.

          \return	void
  */
  virtual void preprocess_image(cv::Mat &raw_color, cv::Mat &raw_depth) = 0;

  //! Preprocess model points
  /*!

  */
  virtual void preprocess_model_points(
      My_Type::Vector3f *dev_model_points,
      My_Type::Vector3f *dev_model_normals) = 0;

  //!
  void copy_previous_intensity_as_model();

 protected:
  //! Filter image
  /*!

  */
  virtual void filter_image(cv::Mat &raw_color, cv::Mat &raw_depth) = 0;

  //! Generate hierarchy image
  virtual void generate_hierarchy_image() = 0;
};

//!
/*!


*/
class Preprocess_RGBD : public Preprocess_engine {
 public:
  //! Default constructor/destructor
  Preprocess_RGBD(){};
  ~Preprocess_RGBD();

  //!
  void init(My_Type::Vector2i raw_color_size, My_Type::Vector2i raw_depth_size,
            int image_alignment_width, int number_of_layers) override;

  //!
  void preprocess_image(cv::Mat &raw_color, cv::Mat &raw_depth) override;

  //!
  void preprocess_model_points(My_Type::Vector3f *dev_model_points,
                               My_Type::Vector3f *dev_model_normals) override;

  //!
  void generate_render_information() override;

 protected:
  //! Filter image
  void filter_image(cv::Mat &raw_color, cv::Mat &raw_depth) override;

  //! Generate hierarchy image
  void generate_hierarchy_image() override;
};
