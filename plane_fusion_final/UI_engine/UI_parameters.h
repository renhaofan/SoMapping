/**
 *  Copyright (C) All rights reserved.
 *  @file UI_parameters.h
 *  @brief Parameters about front UI, viewport size, GLCamera concerned.
 *  @details Parameters include viewport size, window margin width, GLmatrix(v & p matrices) and control sensitivity.
 *  @see UI_parameters.cpp
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include "OurLib/My_vector.h"

/**
 * @brief Singleton of UI_engine parameters.
 * @details Default parameters see set_to_default() in UI_parameters.cpp
 */
class UI_parameters {
 public:
  /** @brief The pointer to this static object. */
  static UI_parameters *instance_ptr;
  /** @brief Member function for instantiating this static object. */
  static UI_parameters *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new UI_parameters();
    return instance_ptr;
  }

  /** @brief Default constructor. call set_to_default() */
  UI_parameters();
  /** @brief Default destructor. Do nothing. */
  ~UI_parameters();

  /** @brief OpenGL windows margin width. */
  int GL_window_margin_width;

  /** @brief Main viewport(3D scene) size. */
  My_Type::Vector2i main_viewport_size;
  /** @brief Sub-viewport(color & depth vis) size. */
  My_Type::Vector2i sub_viewport_size;

  /** @brief View camera rotation sensitivity controlled by keyboard. */
  float UI_camera_rotation_sensitivity;
  /** @brief View camera translation sensitivity controlled by keyboard. */
  float UI_camera_translation_sensitivity;
  /** @brief View camera rotation sensitivity controlled by mouse. */
  float UI_camera_mouse_rotation_sensitivity;
  /** @brief View camera translation sensitivity controlled by mouse. */
  float UI_camera_mouse_translation_sensitivity;

  // Projection matrix concerned.
  /** @brief Minimum, maximum GL camera view distance. */
  My_Type::Vector2f GL_view_range;
  /** @brief OpenGL render aspect. */
  float GL_view_aspect;
  /**
   *  @brief OpenGL view scale for znear, zfar in gluPerspective(__, __, znear, zfar).
   *  @details gluPerspective(__, __, range.x * scale, range.y * scale);
   */
  float GL_view_scale;
  /** @brief OpenGL view camera FOV, Unit in degree. */
  float GL_camera_fov;

  /** @brief Set all paramenters to default value. */
  void set_to_default();
};
