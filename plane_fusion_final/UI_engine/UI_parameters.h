#pragma once

#include "OurLib/My_vector.h"

//!
/*!
        \note	Singleton of UI_engine paramenters
*/
class UI_parameters {
 public:
  //! The pointer to this static object.
  static UI_parameters *instance_ptr;
  //! Member function for instantiating this static object.
  static UI_parameters *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new UI_parameters();
    return instance_ptr;
  }

  //!
  UI_parameters();
  ~UI_parameters();

  //! OpenGL windows margin width
  int GL_window_margin_width;

  //! Main viewport size
  My_Type::Vector2i main_viewport_size;
  //! Sub-viewport size
  My_Type::Vector2i sub_viewport_size;

  //! View camera rotation sensitivity controlled by keyboard.
  float UI_camera_rotation_sensitivity;
  //! View camera translation sensitivity controlled by keyboard.
  float UI_camera_translation_sensitivity;
  //! View camera rotation sensitivity controlled by mouse.
  float UI_camera_mouse_rotation_sensitivity;
  //! View camera translation sensitivity controlled by mouse.
  float UI_camera_mouse_translation_sensitivity;

  //! Minimum, maximum GL camera view distance
  My_Type::Vector2f GL_view_range;
  //! OpenGL render aspect
  float GL_view_aspect;
  //! OpenGL view scale
  float GL_view_scale;
  //! OpenGL view camera FOV
  float GL_camera_fov;

  //! Set all paramenters to default value.
  void set_to_default();
};
