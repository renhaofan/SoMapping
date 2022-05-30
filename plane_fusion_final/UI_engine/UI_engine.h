/**
 *  Copyright (C) All rights reserved.
 *  @file UI_engine.h
 *  @brief jfdk
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @note remove Warning: array subscript has type char.
 *        reference: https://stackoverflow.com/questions/9972359/warning-array-subscript-has-type-char
 *  @todo Make CUDA Timer:
 * https://blog.csdn.net/Morizen/article/details/114265793
 */

#pragma once

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

#include "Data_engine/Data_engine.h"
#include "OurLib/My_frame.h"
#include "OurLib/my_GL_functions.h"
#include "Render_engine/Render_engine.h"
#include "SLAM_system/SLAM_system.h"
#include "UI_engine/UI_parameters.h"

/**
 * @brief The UI_engine class
 */
class UI_engine {
 public:
  /** @brief The pointer to this static object. */
  static UI_engine *instance_ptr;
  /** @brief Member function for instantiating this static object. */
  static UI_engine *instance(void) {
    if (instance_ptr == nullptr) instance_ptr = new UI_engine();
    return instance_ptr;
  }

  /** @brief Data engine to load sequence and write result. */
  Data_engine *data_engine_ptr;
  /** @brief SLAM engine to estimate pose and mapping. */
  SLAM_system *SLAM_system_ptr;
  /** @brief Render engine to draw scene and frame. */
  Render_engine render_engine;

#if __unix__
#pragma region "OpenGL" {
#elif _WIN32
#pragma region(OpenGL)
#endif
  //  GLchar normal_key[256], special_key[256];
  /** @brief Normal key definition for OpenGL keyboard event. */
  uint16_t normal_key[256];
  /** @brief Special key definition for OpenGL keyboard event. */
  uint16_t special_key[256];

  /** @brief Flag whether left mouse button pressed for OpenGL mouse event. */
  bool mouse_LeftDown;
  /** @brief Flag whether right mouse button pressed for OpenGL mouse event. */
  bool mouse_RightDown;
  /** @brief Current mouse position in x direction. */
  GLint mouse_X;
  /** @brief Current mouse position in y direction. */
  GLint mouse_Y;
  /** @brief Position in x direction where the left mouse button pressed. */
  GLint mouse_LeftDown_X;
  /** @brief Position in y direction where the left mouse button pressed. */
  GLint mouse_LeftDown_Y;
  /** @brief Position in x direction where the right mouse button pressed. */
  GLint mouse_RightDown_X;
  /** @brief Position in y direction where the right mouse button pressed. */
  GLint mouse_RightDown_Y;

  /** @brief OpenGL view matrix from camera pose. */
  My_frame<float> GL_camera_Frame;
  /** @brief OpenGL view matrix from camera pose. */
  My_frame<float> GL_camera_Frame_pre;

  /** @brief Reshape flag. */
  bool system_induced_reshape;
  /** @brief Reshape flag. */
  bool need_to_fix_window_aspect;
  /** @brief Reshape flag. */
  float reshape_ratio;

  /** @brief Window width. */
  int window_width;
  /** @brief Window height. */
  int window_height;
  /** @brief Main viewport width. */
  int main_viewport_width;
  /** @brief Main viewport height. */
  int main_viewport_height;
  /** @brief Sub viewport width. */
  int sub_viewport_width;
  /** @brief Sub viewport height. */
  int sub_viewport_height;

  /** @brief PBO(pixel buffer object) for main viewport. */
  GLuint main_viewport_PBO;
  /** @brief Texture for main viewport. */
  GLuint main_viewport_texture;
  /** @brief Texture for sub1 viewport. */
  GLuint sub1_viewport_texture;
  /** @brief Texture for sub2 viewport. */
  GLuint sub2_viewport_texture;

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "CUDA" {
#elif _WIN32
#pragma region(CUDA)
#endif
  /** @brief CUDA timer. */
  StopWatchInterface *timer_instant;
  /** @brief CUDA timer. */
  StopWatchInterface *timer_average;
  /** @brief Elapsed time. */
  float elapsed_time;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Flags" {
#elif _WIN32
#pragma region(Flags)
#endif
  /** @brief Key 0-9 view flags. */
  bool view_flag_list[10];
  /** @brief Draw object of reference. */
  bool show_reference_object;

  /** @brief Render mode of sub-viewport-2. */
  enum SubViewport2Mode {
    /** Pseudo depth image */
    PSEUDO_DEPTH_IMAGE = 0,
    /** Normal image */
    NORMAL_IMAGE = 1,
    /** Residual image */
    RESIDUAL_IMAGE = 2,
    /** Number of enum */
    SVP2MODE_NUM = 3
  };

  /** @brief SubWindow2 render mode, pseudo dpeth image by default. */
  SubViewport2Mode sub_viewport2_mode = SubViewport2Mode::PSEUDO_DEPTH_IMAGE;
  /** @brief Main window render mode, phong render by default. */
  MainViewportRenderMode main_viewport_render_mode =
      MainViewportRenderMode::PHONG_RENDER;

  int render_object_id;
  int render_match_list_id;
  int mesh_render_mode = 0;
  bool record_continuous = false;
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  /** @brief Default constructor. */
  UI_engine();
  /** @brief Default destructor. */
  ~UI_engine();

  /**
   * @brief Initialize flag and status.
   * @param main_argc The number of arguments.
   * @param main_argv The content of arguments.
   * @param data_engine Load sequence and save result.
   * @param SLAM SLAM system to estimate pose and mapping.
   * @param render_engine Render engine to draw scene and frame.
   * @exception data_engine is equal to nullptr.
   * @exception SLAM is equal to nullptr.
   * @exception GlewInit() error.
   * @exception Glew not support GL_VERSION_2_0.
   * @return Status whether init successfully. If exception occured, return -1,
   *         else return 0.
   */
  int init(int main_argc, char **main_argv, Data_engine *data_engine,
           SLAM_system *SLAM, Render_engine *render_engine);

  /** @brief OpenGL main loop. */
  void run();

  /** @brief OpenGL render function, render main window and sub windows. */
  static void OpenGL_DisplayFunction();
  /** @brief OpenGL idle function. */
  static void OpenGL_IdleFunction();
  /** @brief OpenGL reshape function. */
  static void OpenGL_ReshapeFunction(int width, int height);

  /** @brief OpenGL normal key down function. */
  static void OpenGL_NormalKeyFunction(unsigned char key, int x, int y);
  /** @brief OpenGL special key down function. */
  static void OpenGL_SpecialKeyFunction(int key, int x, int y);
  /** @brief OpenGL normal key up function. */
  static void OpenGL_NormalKeyUpFunction(unsigned char key, int x, int y);
  /** @brief OpenGL special key up function. */
  static void OpenGL_SpecialKeyUpFunction(int key, int x, int y);
  /** @brief OpenGL mouse move call back function. */
  static void OpenGL_MouseMoveFunction(int x, int y);
  /** @brief OpenGL mouse button down call back function. */
  static void OpenGL_MouseButtonFunction(int button, int state, int x, int y);
  /** @brief OpenGL mouse wheel call back function. */
  static void OpenGL_MouseWheelFunction(int button, int dir, int x, int y);

 private:
  void interactive_events();

  void fix_window_aspect();

  void render_main_viewport();

  void render_sub_viewport1();

  void render_sub_viewport2();

  void OpenGL_draw_in_OpenGL_camera_coordinate();

  void OpenGL_draw_in_OpenGL_world_coordinate();

  void OpenGL_draw_in_SLAM_world_coordinate();

  void OpenGL_draw_in_some_camera_coordinate(Eigen::Matrix4f camera_pose);
};
