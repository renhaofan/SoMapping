
#include "Main_engine/Main_engine.h"
#include "SLAM_system/SLAM_system.h"
#include "UI_engine/UI_engine.h"

void UI_engine::fix_window_aspect() {
  UI_engine *UI_ptr = UI_engine::instance();

  // Reshape window to fix aspect ratio
  double ratio = (double)UI_ptr->window_height / (double)UI_ptr->window_width;
  double preset_ratio =
      (double)UI_parameters::instance()->main_viewport_size.height /
      (double)(UI_parameters::instance()->main_viewport_size.width +
               UI_parameters::instance()->sub_viewport_size.width);

  // Resize window
  UI_ptr->window_height =
      (int)(0.5f + (double)(UI_ptr->window_width) * preset_ratio);
  glutReshapeWindow(UI_ptr->window_width, UI_ptr->window_height);
  // Update main viewport size
  UI_ptr->main_viewport_width = UI_ptr->window_width * 2.0 / 3.0;
  UI_ptr->main_viewport_height = UI_ptr->window_height;
  // Sub viewport size
  UI_ptr->sub_viewport_width = UI_ptr->main_viewport_width / 2;
  UI_ptr->sub_viewport_height = UI_ptr->main_viewport_height / 2;

  // Render_engine Reshape buffer size
  // My_Type::Vector2i viewport_size(UI_ptr->main_viewport_width,
  // UI_ptr->main_viewport_height);
  // UI_ptr->render_engine.scene_viewport_reshape(viewport_size);
}

//
void UI_engine::interactive_events() {
  Main_engine *main_engine_ptr = Main_engine::instance();

  // Yaw & Pitch
  if (this->special_key[GLUT_KEY_LEFT] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        0.0, -UI_parameters::instance()->UI_camera_rotation_sensitivity, 0.0);
  if (this->special_key[GLUT_KEY_RIGHT] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        0.0, +UI_parameters::instance()->UI_camera_rotation_sensitivity, 0.0);
  if (this->special_key[GLUT_KEY_UP] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        -UI_parameters::instance()->UI_camera_rotation_sensitivity, 0.0, 0.0);
  if (this->special_key[GLUT_KEY_DOWN] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        +UI_parameters::instance()->UI_camera_rotation_sensitivity, 0.0, 0.0);
  // (Normal key)
  if (this->normal_key['d'] == 1 || this->normal_key['D'] == 1)
    this->GL_camera_Frame.transfer_inFrame(
        -UI_parameters::instance()->UI_camera_translation_sensitivity, 0, 0);
  if (this->normal_key['a'] == 1 || this->normal_key['A'] == 1)
    this->GL_camera_Frame.transfer_inFrame(
        +UI_parameters::instance()->UI_camera_translation_sensitivity, 0, 0);
  if (this->normal_key['w'] == 1 || this->normal_key['W'] == 1)
    this->GL_camera_Frame.transfer_inFrame(
        0, 0, +UI_parameters::instance()->UI_camera_translation_sensitivity);
  if (this->normal_key['s'] == 1 || this->normal_key['S'] == 1)
    this->GL_camera_Frame.transfer_inFrame(
        0, 0, -UI_parameters::instance()->UI_camera_translation_sensitivity);

  // Roll
  if (this->special_key[GLUT_KEY_PAGE_UP] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        0.0, 0.0, -UI_parameters::instance()->UI_camera_rotation_sensitivity);
  if (this->special_key[GLUT_KEY_PAGE_DOWN] == 1)
    this->GL_camera_Frame.rotate_inFrame(
        0.0, 0.0, +UI_parameters::instance()->UI_camera_rotation_sensitivity);

  // zoom-in
  if ((this->normal_key['e'] == 1 || this->normal_key['E'] == 1) &&
      (this->GL_camera_Frame.view_distance > 0)) {
    this->GL_camera_Frame.view_distance -=
        UI_parameters::instance()->UI_camera_translation_sensitivity;
    this->GL_camera_Frame.transfer_inFrame(
        0, 0, +UI_parameters::instance()->UI_camera_translation_sensitivity);
  }
  // zoom-out
  if (this->normal_key['q'] == 1 || this->normal_key['Q'] == 1) {
    this->GL_camera_Frame.view_distance +=
        UI_parameters::instance()->UI_camera_translation_sensitivity;
    this->GL_camera_Frame.transfer_inFrame(
        0, 0, -UI_parameters::instance()->UI_camera_translation_sensitivity);
  }

  // Rotation (mouse)
  if (this->mouse_LeftDown) {
    float d_x, d_y;
    d_x = (float)(this->mouse_LeftDown_X - this->mouse_X) *
          UI_parameters::instance()->UI_camera_mouse_rotation_sensitivity;
    d_y = (float)(this->mouse_LeftDown_Y - this->mouse_Y) *
          UI_parameters::instance()->UI_camera_mouse_rotation_sensitivity;

    this->GL_camera_Frame.load_frame_mat(this->GL_camera_Frame_pre);
    this->GL_camera_Frame.rotate_inFrame(-d_y, -d_x, 0.0);
  }

  // Translation (mouse)
  if (this->mouse_RightDown) {
    float d_x, d_y;
    d_x = (float)(this->mouse_RightDown_X - this->mouse_X) *
          UI_parameters::instance()->UI_camera_mouse_translation_sensitivity;
    d_y = (float)(this->mouse_RightDown_Y - this->mouse_Y) *
          UI_parameters::instance()->UI_camera_mouse_translation_sensitivity;

    this->GL_camera_Frame.load_frame_mat(this->GL_camera_Frame_pre);
    this->GL_camera_Frame.transfer_inWrold(0.0, +d_y, 0.0);
    // this->GL_camera_Frame.transfer_inFrame(-d_x, 0.0, 0.0);
  }

  // Continuous image processing
  if (this->normal_key['c'] == 1 || this->normal_key['C'] == 1)
    this->SLAM_system_ptr->processing_state =
        ProcessingState::PROCESS_CONTINUOUS_FRAME;
  // Single frame image processing
  if (this->normal_key['f'] == 1 || this->normal_key['F'] == 1)
    this->SLAM_system_ptr->processing_state =
        ProcessingState::PROCESS_SINGLE_FRAME;

  if (this->normal_key['p'] == 1 || this->normal_key['P'] == 1) {
  }

  // debug coeff
  if (this->normal_key['['] == 1 || this->normal_key['{'] == 1) {
  }
  if (this->normal_key[']'] == 1 || this->normal_key['}'] == 1) {
  }
}
