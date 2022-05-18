#pragma once

// IO
#include <cstdio>
#include <functional>
#include <iostream>
using namespace std;

// OpenGL header files
// glew
#include <GL/glew.h>
// freeglut
#include <GL/freeglut.h>
#include <GL/glut.h>

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//
#include "Data_engine/Data_engine.h"
#include "Render_engine/Render_engine.h"
#include "SLAM_system/SLAM_system.h"
//
#include "OurLib/My_frame.h"
#include "OurLib/my_GL_functions.h"
#include "UI_engine/UI_parameters.h"

//!
/*!


*/
class UI_engine {
 public:
  //! The pointer to this static object.
  static UI_engine *instance_ptr;
  //! Member function for instantiating this static object.
  static UI_engine *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new UI_engine();
    return instance_ptr;
  }

  //!
  Data_engine *data_engine_ptr;
  //!
  SLAM_system *SLAM_system_ptr;

  //!
  Render_engine render_engine;

#pragma region(OpenGL)
  //!
  GLchar normal_key[256], special_key[256];
  //!
  bool mouse_LeftDown, mouse_RightDown;
  GLint mouse_X, mouse_Y;
  GLint mouse_LeftDown_X, mouse_LeftDown_Y;
  GLint mouse_RightDown_X, mouse_RightDown_Y;

  // OpenGL view camera pose
  My_frame<float> GL_camera_Frame, GL_camera_Frame_pre;

  //! Reshape flag
  bool system_induced_reshape;
  bool need_to_fix_window_aspect;
  float reshape_ratio;
  // windows size
  int window_width, window_height;
  int main_viewport_width, main_viewport_height;
  int sub_viewport_width, sub_viewport_height;

  // PBO, Texture
  GLuint main_viewport_PBO;
  GLuint main_viewport_texture, sub1_viewport_texture, sub2_viewport_texture;

#pragma endregion

#pragma region(CUDA)
  // Timer
  StopWatchInterface *timer_instant;
  StopWatchInterface *timer_average;
  float elapsed_time;

#pragma endregion

#pragma region(Flags)

  // View flags
  bool view_flag_list[10];
  // Draw object of reference
  bool show_reference_object;
  // Render mode of Sub-viewport-2
  enum SubViewport2Mode {
    PSEUDO_DEPTH_IMAGE = 0,
    NORMAL_IMAGE = 1,
    RESIDUAL_IMAGE = 2,
    // GRADIENT_IMAGE = 3,
  };
  SubViewport2Mode sub_viewport2_mode = SubViewport2Mode::PSEUDO_DEPTH_IMAGE;
  // Render mode of main viewport
  MainViewportRenderMode main_viewport_render_mode =
      MainViewportRenderMode::PHONG_RENDER;
  //
  int render_object_id;
  //
  int render_match_list_id;
  //
  int mesh_render_mode = 0;

  //
  bool record_continuous = false;

#pragma endregion

  //! Default constructor/destructor
  UI_engine();
  ~UI_engine();

  // Initiation
  int init(int main_argc, char **main_argv, Data_engine *data_engine,
           SLAM_system *SLAM_system, Render_engine *render_engine);

  // OpenGL main loop
  void run();

  //! OpenGL render function
  static void OpenGL_DisplayFunction();
  //! OpenGL idle function
  static void OpenGL_IdleFunction();
  //! OpenGL reshape function
  static void OpenGL_ReshapeFunction(int width, int height);
  //! OpenGL normal key down function
  static void OpenGL_NormalKeyFunction(unsigned char key, int x, int y);
  //! OpenGL special key down function
  static void OpenGL_SpecialKeyFunction(int key, int x, int y);
  //! OpenGL normal key up function
  static void OpenGL_NormalKeyUpFunction(unsigned char key, int x, int y);
  //! OpenGL special key up function
  static void OpenGL_SpecialKeyUpFunction(int key, int x, int y);
  //! OpenGL mouse move call back function
  static void OpenGL_MouseMoveFunction(int x, int y);
  //! OpenGL mouse button down call back function
  static void OpenGL_MouseButtonFunction(int button, int state, int x, int y);
  //! OpenGL mouse wheel call back function
  static void OpenGL_MouseWheelFunction(int button, int dir, int x, int y);

 private:
  //!
  void interactive_events();

  //!
  void fix_window_aspect();

  //!
  void render_main_viewport();

  //!
  void render_sub_viewport1();

  //!
  void render_sub_viewport2();

  //!
  void OpenGL_draw_in_OpenGL_camera_coordinate();

  //
  void OpenGL_draw_in_OpenGL_world_coordinate();

  //!
  void OpenGL_draw_in_SLAM_world_coordinate();

  //!
  void OpenGL_draw_in_some_camera_coordinate(Eigen::Matrix4f camera_pose);
};
