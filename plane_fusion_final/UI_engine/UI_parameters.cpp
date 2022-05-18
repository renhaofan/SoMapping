#include "UI_parameters.h"

//
UI_parameters *UI_parameters::instance_ptr = nullptr;

//
UI_parameters::UI_parameters() { this->set_to_default(); }
UI_parameters::~UI_parameters() {}

//!
void UI_parameters::set_to_default() {
  //
  this->GL_window_margin_width = 8;
  //
  this->main_viewport_size.width = 640;
  this->main_viewport_size.height = 480;
  //
  this->sub_viewport_size.width = 320;
  this->sub_viewport_size.height = 240;
  //
  this->UI_camera_rotation_sensitivity = 0.003f;
  this->UI_camera_translation_sensitivity = 0.01f;
  this->UI_camera_mouse_rotation_sensitivity = 0.01f;
  this->UI_camera_mouse_translation_sensitivity = 0.005f;
  //
  this->GL_view_range.x = 0.01f;
  this->GL_view_range.y = 100.0f;
  //
  this->GL_view_aspect = 1.33333f;
  this->GL_view_scale = 1.0f;
  this->GL_camera_fov = 53.1301f; /*! FOV = 2 * atan(cy/fy) = 2 * atan(240/480)
                                   = 53.1301 Degree */
}
