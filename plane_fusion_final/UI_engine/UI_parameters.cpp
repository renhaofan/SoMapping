/**
 *  Copyright (C) All rights reserved.
 *  @file UI_parameters.cpp
 *  @brief Implement the functon: UI_parameters::set_to_default() where default
 *         UI parameters are defined.
 *  @details Main window size: 640*480.
 *  @details Sub window size: 320*240.
 *  @details Margin width: 8.
 *  @details UI_camera_rotation_sensitivity = 0.003f;
 *  @details UI_camera_translation_sensitivity = 0.01f;
 *  @details UI_camera_mouse_rotation_sensitivity = 0.01f;
 *  @details UI_camera_mouse_translation_sensitivity = 0.005f;
 *  @details Projection matrix: 0.01*scale-100*scale(near-far), scale 1.f,
 *           aspect 1.333f(640/480), fov 53.1301f degree.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#include "UI_parameters.h"

UI_parameters *UI_parameters::instance_ptr = nullptr;

UI_parameters::UI_parameters() { this->set_to_default(); }
UI_parameters::~UI_parameters() {}

void UI_parameters::set_to_default() {
  this->GL_window_margin_width = 8;

  this->main_viewport_size.width = 640;
  this->main_viewport_size.height = 480;

  this->sub_viewport_size.width = 320;
  this->sub_viewport_size.height = 240;

  this->UI_camera_rotation_sensitivity = 0.003f;
  this->UI_camera_translation_sensitivity = 0.01f;
  this->UI_camera_mouse_rotation_sensitivity = 0.01f;
  this->UI_camera_mouse_translation_sensitivity = 0.005f;

  this->GL_view_range.x = 0.01f;
  this->GL_view_range.y = 100.0f;

  // 640/480
  this->GL_view_aspect = 1.33333f;
  this->GL_view_scale = 1.0f;
  // FOV = 2 * atan(cy/fy) = 2 * atan(240/480)  = 53.1301 Degree.
  this->GL_camera_fov = 53.1301f;
}
