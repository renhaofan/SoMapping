/**
 *  @file Preprocess_engine.h
 *  @brief Frame filter and frame hierarchy generation.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @todo void Preprocess_RGBD::preprocess_image() bilateral filter BUGS.
 */

#pragma once

#include <cv.h>
#include <float.h>
#include <highgui.h>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "Hierarchy_image.h"
#include "OurLib/My_matrix.h"
#include "Preprocess_KernelFunc.cuh"
#include "SLAM_system/SLAM_system_settings.h"

class Preprocess_engine {
 public:
  /** @brief Raw color image size. */
  My_Type::Vector2i raw_color_size;
  /** @brief Raw depth image size. */
  My_Type::Vector2i raw_depth_size;

  /**
   *  @brief Image width alignemnt. Refine hierarrchy image layer size, used by
   function ceil_by_stride().
   *  @see void init_parameters(My_Type::Vector2i image_size, int
           alignment_size, int layer_number) in Hierarchy_image.h
   */
  int image_alignment_width;

  /**
   *  @brief Depth size from function ceil_by_stride(raw_depth_size,
             alignment)
   */
  My_Type::Vector2i aligned_depth_size;

  // <------------------ Host ------------------
  // Depth information
  /** @brief Host variable,  */
  My_Type::Vector3f *raw_aligned_points;
  /** @brief Host variable, points generated from filtered points. */
  Hierarchy_image<My_Type::Vector3f> hierarchy_points;
  /** @brief Host variable, normal vectors computed from filtered depth. */
  Hierarchy_image<My_Type::Vector3f> hierarchy_normals;
  /** @brief Host variable, points generated from model. */
  Hierarchy_image<My_Type::Vector3f> hierarchy_model_points;
  /** @brief Host variable, normal vectors computed from model. */
  Hierarchy_image<My_Type::Vector3f> hierarchy_model_normals;
  // Intensity information
  /** @brief Host variable,  */
  Hierarchy_image<float> hierarchy_intensity;
  /** @brief Host variable,  */
  Hierarchy_image<My_Type::Vector2f> hierarchy_gradient;
  /** @brief Host variable,  */
  Hierarchy_image<float> hierarchy_model_intensity;
  /** @brief Host variable,  */
  Hierarchy_image<My_Type::Vector2f> hierarchy_model_gradient;
  // ------------------ Host ------------------>

  // <------------------ Device ------------------
  // Depth information
  /** @brief Device pointer of depth image (unsigned short). */
  RawDepthType *dev_raw_depth;
  /** @brief Device pointer of float-type depth image buffer. */
  float *dev_depth_buffer;
  /** @brief Device pointer of float-type filterd depth image buffer. */
  float *dev_filtered_depth;

  /** @brief Device pointer of raw aligned points. */
  My_Type::Vector3f *dev_raw_aligned_points;

  /** @brief Hierarchy image records points(xyz). */
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_points;
  /** @brief Hierarchy image records points' normals.(xyz). */
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_normals;

  /** @brief Hierarchy image records points generated from model. */
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_model_points;
  /** @brief Hierarchy image records points' normals computed from model. */
  Hierarchy_image<My_Type::Vector3f> dev_hierarchy_model_normals;

  // Intensity information
  /** @brief Device pointer of Color image (RGB). */
  RawColorType *dev_raw_color;
  /** @brief Hierarchy insensity(gray) image. */
  Hierarchy_image<float> dev_hierarchy_intensity;
  /** @brief Hierarchy image records 2D intensity gradient. */
  Hierarchy_image<My_Type::Vector2f> dev_hierarchy_gradient;
  /** @brief Not sure??????. */
  Hierarchy_image<float> dev_hierarchy_model_intensity;
  /** @brief Not sure??????. */
  Hierarchy_image<My_Type::Vector2f> dev_hierarchy_model_gradient;
  // ------------------ Device ------------------>

  virtual void generate_render_information() = 0;

  //! Default constructor/destructor
  virtual ~Preprocess_engine(){};

  /**
   * @brief Initialization
   * @param raw_color_size Raw color image size.
   * @param raw_depth_size Raw depth image size.
   * @param image_alignment_width Image alignment width size, used by function
   *        ceil_by_stride(_, image_alignment_width)
   * @param number_of_layers The number of hierarchy layers.
   */
  virtual void init(My_Type::Vector2i raw_color_size,
                    My_Type::Vector2i raw_depth_size, int image_alignment_width,
                    int number_of_layers);

  /**
   * @brief preprocess_image
   * @param raw_color Raw color image captured by sensor.
   * @param raw_depth Raw depth image captured by sensor.
   */
  virtual void preprocess_image(cv::Mat &raw_color, cv::Mat &raw_depth) = 0;

  /**
   * @brief Preprocess model points
   * @param dev_model_points
   * @param dev_model_normals
   */
  virtual void preprocess_model_points(
      My_Type::Vector3f *dev_model_points,
      My_Type::Vector3f *dev_model_normals) = 0;

  /**
   * @brief copy_previous_intensity_as_model
   */
  void copy_previous_intensity_as_model();

 protected:
  /**
   * @brief Filter image
   * @param raw_color
   * @param raw_depth
   */
  virtual void filter_image(cv::Mat &raw_color, cv::Mat &raw_depth) = 0;

  /**
   * @brief Generate hierarchy image
   */
  virtual void generate_hierarchy_image() = 0;
};

class Preprocess_RGBD : public Preprocess_engine {
 public:
  /** @brief Default constructor. */
  Preprocess_RGBD(){};
  /** @brief Default destructor. */
  ~Preprocess_RGBD();

  /**
   * @brief Allocate memory for hierarchy images.
   * @param raw_color_size Raw color image size.
   * @param raw_depth_size Raw depth image size.
   * @param image_alignment_widt image_alignment_width Image alignment width
   * size, used by function ceil_by_stride(_, image_alignment_width)
   * @param number_of_layers The number of hierarchy layers.
   */
  void init(My_Type::Vector2i raw_color_size, My_Type::Vector2i raw_depth_size,
            int image_alignment_width, int number_of_layers) override;

  /**
   * @brief Frame filter and hierarchy generation.
   * @todo BUG to be solved.
   * @param raw_color Raw color image. (cv::Mat)
   * @param raw_depth Raw depth image. (cv::Mat)
   */
  void preprocess_image(cv::Mat &raw_color, cv::Mat &raw_depth) override;

  /**
   * @brief Generate hierarchy depth and normal image.
   * @param dev_model_points Model points.
   * @param dev_model_normals Model normals.
   */
  void preprocess_model_points(My_Type::Vector3f *dev_model_points,
                               My_Type::Vector3f *dev_model_normals) override;

  void generate_render_information() override;

 protected:
  /**
   * @brief Bilateral filter for depth image and generate intensity image for
   *        camera pose tracking.
   * @param raw_color Raw color image. (cv::Mat)
   * @param raw_depth Raw depth image. (cv::Mat)
   */
  void filter_image(cv::Mat &raw_color, cv::Mat &raw_depth) override;

  /** @brief Generate hierarchy image. */
  void generate_hierarchy_image() override;
};
