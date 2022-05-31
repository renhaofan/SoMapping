/**
 *  Copyright (C) All rights reserved.
 *  @file UI_engine.cpp
 *  @brief Implement main function defined in UI_engine.h except
 *         UI_engine::fix_window_aspect() and UI_engine::interactive_events().
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#include "UI_engine.h"

#include "OurLib/my_GL_functions.h"
#include "OurLib/my_GL_geometry.h"

// using std::to_string()
#include <iostream>
#include <string>

UI_engine *UI_engine::instance_ptr = nullptr;

void screen_shot(string &save_folder, int frame_id);

UI_engine::UI_engine() {
  // Initialize flags
  this->mouse_LeftDown = false;
  this->mouse_RightDown = false;
  this->show_reference_object = false;

  this->system_induced_reshape = false;
  this->need_to_fix_window_aspect = false;
  this->reshape_ratio = 1.0f;

  for (int i = 0; i < 10; i++) this->view_flag_list[i] = true;
  this->view_flag_list[0] = false;
  this->view_flag_list[1] = false;
  this->view_flag_list[2] = false;
  this->view_flag_list[3] = false;
  this->view_flag_list[4] = false;
  this->view_flag_list[5] = false;
  this->view_flag_list[6] = false;
  this->view_flag_list[7] = false;
  // this->view_flag_list[8] = false;
  this->view_flag_list[9] = false;

  this->sub_viewport2_mode = SubViewport2Mode::PSEUDO_DEPTH_IMAGE;
  this->main_viewport_render_mode = MainViewportRenderMode::PHONG_RENDER;
  this->render_object_id = 1;
  this->render_match_list_id = 0;

  this->data_engine_ptr = nullptr;
}
UI_engine::~UI_engine() {}

int UI_engine::init(int main_argc, char **main_argv, Data_engine *data_engine,
                    SLAM_system *SLAM_system, Render_engine *render_engine) {
#if __unix__
#pragma region "Get module pointer" {
#elif _WIN32
#pragma region(Get module pointer)
#endif

  // Get data_engine
  if (data_engine != nullptr) {
    this->data_engine_ptr = data_engine;
  } else {
#ifdef LOGGING
    LOG_FATAL("UI_Engine error: invlid data_engine_ptr!");
    Log::shutdown();
#endif
    fprintf(stderr, "UI_Engine error: invlid data_engine_ptr !\r\n");
    exit(1);
  }

  // Get SLAM_system
  if (SLAM_system != nullptr) {
    this->SLAM_system_ptr = SLAM_system;
  } else {
#ifdef LOGGING
    LOG_FATAL("UI_Engine error : invlid SLAM_system_ptr!");
    Log::shutdown();
#endif
    fprintf(stderr, "UI_Engine error : invlid SLAM_system_ptr !\r\n");
    exit(1);
  }

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  // Set window size
  this->window_width = UI_parameters::instance()->main_viewport_size.width +
                       UI_parameters::instance()->sub_viewport_size.width +
                       UI_parameters::instance()->GL_window_margin_width;
  this->window_height = UI_parameters::instance()->main_viewport_size.height +
                        UI_parameters::instance()->GL_window_margin_width;
  // Set viewport size
  this->main_viewport_width =
      UI_parameters::instance()->main_viewport_size.width;
  this->main_viewport_height =
      UI_parameters::instance()->main_viewport_size.height;
  this->sub_viewport_width = this->main_viewport_width / 2;
  this->sub_viewport_height = this->main_viewport_height / 2;

  // Initialize render engine
  //  My_Type::Vector2i scene_viewport_size(this->main_viewport_width,
  //                                        this->main_viewport_height);
  My_Type::Vector2i depth_size;
  this->data_engine_ptr->get_depth_image_size(depth_size.width,
                                              depth_size.height);
  this->render_engine.init(depth_size, depth_size);

#if __unix__
#pragma region "OpenGL Initialization" {
#elif _WIN32
#pragma region(OpenGL Initialization)
#endif

  // OpenGL initialization
  glutInit(&main_argc, main_argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(this->window_width, this->window_height);
  glutCreateWindow("My SLAM System - base line");
  // Register OpenGL display function
  glutDisplayFunc(OpenGL_DisplayFunction);
  // Register OpenGL idle function
  glutIdleFunc(OpenGL_IdleFunction);
  // Register OpenGL normal keypress callback function
  glutKeyboardUpFunc(OpenGL_NormalKeyUpFunction);
  // Register OpenGL special keypress callback function
  glutSpecialUpFunc(OpenGL_SpecialKeyUpFunction);
  // Register OpenGL normal key up callback function
  glutKeyboardFunc(OpenGL_NormalKeyFunction);
  // Register OpenGL special key up callback function
  glutSpecialFunc(OpenGL_SpecialKeyFunction);
  // Register OpenGL mouse move callback function
  glutMotionFunc(OpenGL_MouseMoveFunction);
  // Register OpenGL mouse button press callback function
  glutMouseFunc(OpenGL_MouseButtonFunction);
  // Register OpenGL mouse wheel press callback function
  glutMouseWheelFunc(OpenGL_MouseWheelFunction);
  // Register OpenGL windows reshape callback function
  glutReshapeFunc(OpenGL_ReshapeFunction);

  // Initialize GLEW environment (must do this after initialization of glut)
  GLenum err = glewInit();
  if (err != GLEW_OK) {
#ifdef LOGGING
    LOG_FATAL(glewGetErrorString(err));
    Log::shutdown();
#endif
    fprintf(stderr, "%s\n", glewGetErrorString(err));
    return -1;
  }
  if (!glewIsSupported("GL_VERSION_2_0")) {
#ifdef LOGGING
    LOG_FATAL("Support for necessary OpengGL extensions missing.");
    Log::shutdown();
#endif
    fprintf(stderr, "ERROR: Support for necessary OpengGL extensions missing.");
    return -1;
  }

  // Initialization key press state
  memset(normal_key, 0, 256 * sizeof(GLchar));
  memset(special_key, 0, 256 * sizeof(GLchar));

  int width, height;
  // Generate PBO
  glGenBuffers(1, &(this->main_viewport_PBO));
  glBindBuffer(GL_ARRAY_BUFFER, this->main_viewport_PBO);
  glBufferData(GL_ARRAY_BUFFER,
               UI_parameters::instance()->main_viewport_size.width *
                   UI_parameters::instance()->main_viewport_size.height * 3,
               NULL, GL_STREAM_COPY);
  glBindBuffer(GL_ARRAY_BUFFER, GLuint(NULL));
  // Generate Texture
  glGenTextures(1, &(this->main_viewport_texture));
  glBindTexture(GL_TEXTURE_2D, this->main_viewport_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
               UI_parameters::instance()->main_viewport_size.width,
               UI_parameters::instance()->main_viewport_size.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, GLuint(NULL));

  this->data_engine_ptr->get_color_image_size(width, height);
  glGenTextures(1, &(this->sub1_viewport_texture));
  glBindTexture(GL_TEXTURE_2D, this->sub1_viewport_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, GLuint(NULL));

  this->data_engine_ptr->get_depth_image_size(width, height);
  glGenTextures(1, &(this->sub2_viewport_texture));
  glBindTexture(GL_TEXTURE_2D, this->sub2_viewport_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, GLuint(NULL));

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

  // Initialize GL camera view pose
  this->GL_camera_Frame.set_Identity();
  this->GL_camera_Frame_pre.set_Identity();

  // Initialize CUDA timer
  sdkCreateTimer(&timer_instant);
  sdkCreateTimer(&timer_average);
  sdkResetTimer(&timer_average);

  return 0;
}

void UI_engine::run() { glutMainLoop(); }

void UI_engine::OpenGL_DisplayFunction() {
  UI_engine *UI_ptr = UI_engine::instance();

  // Prepare for rendering. Clear OpenGL render buffers.
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  // Render main viewport
  UI_ptr->render_main_viewport();

  // Render sub viewport_1
  UI_ptr->render_sub_viewport1();

  // Render sub viewport_2
  UI_ptr->render_sub_viewport2();

  // Display on screen
  glutSwapBuffers();
}

void UI_engine::OpenGL_IdleFunction() {
  UI_engine *UI_ptr = UI_engine::instance();

  // Reshape window to fix aspect ratio
  if (UI_ptr->need_to_fix_window_aspect) {
    UI_ptr->need_to_fix_window_aspect = false;
    UI_ptr->system_induced_reshape = true;
    UI_ptr->fix_window_aspect();
  }

  // Inspect keyboard state
  UI_ptr->interactive_events();

  // Draw model surface
  if (UI_ptr->view_flag_list[7])
    UI_ptr->SLAM_system_ptr->need_generate_mesh = true;
  else
    UI_ptr->SLAM_system_ptr->need_generate_mesh = false;

  ProcessingState current_state = UI_ptr->SLAM_system_ptr->process_frames();
  // Check status, if necessary.
  if (current_state) {
  }

  if (UI_ptr->record_continuous && UI_ptr->SLAM_system_ptr->processing_state !=
                                       ProcessingState::STOP_PROCESS) {
    if (false) {
      Eigen::Matrix4f current_pose_mat =
          UI_ptr->SLAM_system_ptr->estimated_camera_pose.mat.eval();

      // Coordinate transformation
      Eigen::Matrix4f coordinate_change;
      coordinate_change.setIdentity();
      // coordinate_change(0, 0) = -1;
      coordinate_change(1, 1) = -1;
      coordinate_change(2, 2) = -1;

      current_pose_mat.block(0, 0, 3, 3) =
          coordinate_change * current_pose_mat.block(0, 0, 3, 3).eval() *
          coordinate_change.inverse();
      current_pose_mat.block(0, 3, 3, 1) =
          coordinate_change * current_pose_mat.block(0, 3, 3, 1).eval();
      current_pose_mat = current_pose_mat.inverse();

      float *ptr_f1 = UI_ptr->GL_camera_Frame.mat.data;
      //

      float *ptr_f2 = current_pose_mat.data();
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          ptr_f1[i * 4 + j] = ptr_f2[i * 4 + j];
        }
      }
    }

    // Screen shot counter
    static int screen_shot_counter = 0;
    string screen_shot_path = "main_viewport_screenshot";

    //
    screen_shot(screen_shot_path, screen_shot_counter);

    // to next shot counter
    screen_shot_counter++;
  }

  glutPostRedisplay();
}

void UI_engine::OpenGL_ReshapeFunction(int width, int height) {
  UI_engine *UI_ptr = UI_engine::instance();

  if (UI_ptr->system_induced_reshape) {
    UI_ptr->system_induced_reshape = false;
    UI_ptr->need_to_fix_window_aspect = false;
  } else {
    UI_ptr->need_to_fix_window_aspect = true;
  }

  UI_ptr->window_width = width;
  UI_ptr->window_height = height;
  UI_ptr->reshape_ratio =
      (float)(width) /
      (float)(UI_parameters::instance()->main_viewport_size.width +
              UI_parameters::instance()->sub_viewport_size.width);
  //  if (0.01f < fabs((float)(width) / (float)(height)-(float)(MAIN_VIEWPORT_W
  //  + SUB_VIEWPORT_W) / (float)(MAIN_VIEWPORT_H))) printf("Reshape : (%d,
  //  %d)\r\n", width, height);

#ifdef LOGGING
  LOG_INFO("Call OpenGL reshape function, windows size(w, h): (" +
           to_string(width) + ", " + to_string(height) + ")");
#endif
}

void UI_engine::render_main_viewport() {
  // CUDA render scene
  {
    Eigen::Matrix4f GL_view_pose;
    float *ptr_f = GL_view_pose.data();
    if (true) {
      for (int i = 0; i < 16; i++) ptr_f[i] = this->GL_camera_Frame.mat[i];
    } else {
      GL_view_pose.setIdentity();
    }
    // Note : OpenGL view pose is the inverse of OpenGL transformation matrix !
    GL_view_pose = GL_view_pose.inverse().eval();

    //
    Eigen::Matrix3f coordinate_change;
    coordinate_change.setIdentity();
    coordinate_change(1, 1) = -1;
    coordinate_change(2, 2) = -1;
    GL_view_pose.block(0, 0, 3, 3) = coordinate_change *
                                     GL_view_pose.eval().block(0, 0, 3, 3) *
                                     coordinate_change.inverse();
    GL_view_pose.block(0, 3, 3, 1) =
        coordinate_change * GL_view_pose.eval().block(0, 3, 3, 1);

    // this->SLAM_system_ptr->generate_render_info(GL_view_pose);
    this->SLAM_system_ptr->preprocess_engine->generate_render_information();
  }

  glViewport(0, 0, this->main_viewport_width, this->main_viewport_height);

  // Projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(UI_parameters::instance()->GL_camera_fov,
                 UI_parameters::instance()->GL_view_aspect,
                 UI_parameters::instance()->GL_view_range.x *
                     UI_parameters::instance()->GL_view_scale,
                 UI_parameters::instance()->GL_view_range.y *
                     UI_parameters::instance()->GL_view_scale);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Enable OpenGL Anti-aliasing
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_MULTISAMPLE);
  // Enable OpenGL blend mode
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw object of refernce
  if ((this->GL_camera_Frame.view_distance >
       0.03f * UI_parameters::instance()->GL_view_scale) &&
      (this->show_reference_object)) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor4f(0.618f, 0.618f, 0.618f, 0.618f);
    glTranslatef(0, 0, -this->GL_camera_Frame.view_distance);
    glutSolidSphere(0.02 * UI_parameters::instance()->GL_view_scale, 16, 16);
    glTranslatef(0, 0, +this->GL_camera_Frame.view_distance);
  }

  // 1. Draw OpenGL world coordiate
  if (this->view_flag_list[1]) {
    // Transform to OpenGL world coordinate
    this->OpenGL_draw_in_OpenGL_world_coordinate();

    // Set draw color
    glColor4f(1.0, 1.0, 1.0, 1.0);
    draw_coordinate_GL(0.5 * UI_parameters::instance()->GL_view_scale, 3.0);
  }

  // 2. Draw camera trajectories
  if (this->view_flag_list[2]) {
    // Transform to SLAM world coordinate
    this->OpenGL_draw_in_SLAM_world_coordinate();

    // Draw ground truth trajectory
    if (true) {
      Trajectory ground_truth_trajectory =
          this->data_engine_ptr->get_ground_truth_trajectory();
      //
      glColor4f(0.5, 0.5, 1.0, 0.5);
      glLineWidth(5.0f);
      glBegin(GL_LINES);
      for (int trajectory_id = 1;
           trajectory_id < min(this->SLAM_system_ptr->frame_id,
                               (int)ground_truth_trajectory.size());
           trajectory_id++) {
        Trajectory_node temp_node;
        temp_node = ground_truth_trajectory[trajectory_id - 1];
        glVertex3f(temp_node.tx * UI_parameters::instance()->GL_view_scale,
                   temp_node.ty * UI_parameters::instance()->GL_view_scale,
                   temp_node.tz * UI_parameters::instance()->GL_view_scale);
        temp_node = ground_truth_trajectory[trajectory_id];
        glVertex3f(temp_node.tx * UI_parameters::instance()->GL_view_scale,
                   temp_node.ty * UI_parameters::instance()->GL_view_scale,
                   temp_node.tz * UI_parameters::instance()->GL_view_scale);
      }
      glEnd();
    }

    // Draw estimated camera pose
    if (true) {
      Trajectory &estimated_trajectory =
          this->SLAM_system_ptr->estimated_trajectory;
      //
      glColor4f(1.0, 0.5, 0.5, 0.5);
      glLineWidth(5.0f);
      glBegin(GL_LINES);
      for (int trajectory_id = 1;
           trajectory_id < min(this->SLAM_system_ptr->frame_id,
                               (int)estimated_trajectory.size());
           trajectory_id++) {
        Trajectory_node temp_node;
        temp_node = estimated_trajectory[trajectory_id - 1];
        glVertex3f(temp_node.tx * UI_parameters::instance()->GL_view_scale,
                   temp_node.ty * UI_parameters::instance()->GL_view_scale,
                   temp_node.tz * UI_parameters::instance()->GL_view_scale);
        temp_node = estimated_trajectory[trajectory_id];
        glVertex3f(temp_node.tx * UI_parameters::instance()->GL_view_scale,
                   temp_node.ty * UI_parameters::instance()->GL_view_scale,
                   temp_node.tz * UI_parameters::instance()->GL_view_scale);
      }
      glEnd();
    }
  }

  // 3. Draw voxel block
  if (this->view_flag_list[3]) {
    // if (typeid(this->SLAM_system_ptr->map_engine) == typeid(Basic_Voxel_map
    // *))
    //{
    //	//this->render_engine.generate_voxel_block_lines();
    //}
    // if (typeid(this->SLAM_system_ptr->map_engine) ==
    // typeid(Submap_Voxel_map))
    //{
    //}
    if (false) {
      this->OpenGL_draw_in_SLAM_world_coordinate();
      //
      Submap_SLAM_system *sys_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      Submap_Voxel_map *map_ptr =
          dynamic_cast<Submap_Voxel_map *>(sys_ptr->submap_ptr_array.back());
      this->render_engine.generate_voxel_block_lines(
          map_ptr->voxel_map_ptr->dev_entrise);
      // printf("this->render_engine.number_of_blocks = %d\n",
      // this->render_engine.number_of_blocks);
      //
      glLineWidth(1.0f);
      glColor4f(1.0f, 1.0f, 0.0f, 0.7f);
      glEnableClientState(GL_VERTEX_ARRAY);

      glVertexPointer(3, GL_FLOAT, sizeof(My_Type::Vector3f),
                      this->render_engine.voxel_block_lines);
      glDrawArrays(GL_LINES, 0, this->render_engine.number_of_blocks * 24);

      glDisableClientState(GL_VERTEX_ARRAY);
    }

    // Debug non-planar region
    if (true) {
      this->OpenGL_draw_in_SLAM_world_coordinate();
      //
      Submap_SLAM_system *sys_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      Submap_Voxel_map *map_ptr =
          dynamic_cast<Submap_Voxel_map *>(sys_ptr->submap_ptr_array.back());
      //
      sys_ptr->mesh_ptr_array.back()->generate_nonplanar_mesh_from_voxel(
          map_ptr->voxel_map_ptr->dev_entrise,
          map_ptr->voxel_map_ptr->dev_voxel_block_array);
      //
      this->render_engine.generate_voxel_block_lines(
          sys_ptr->mesh_ptr_array.back()->dev_nonplanar_entries,
          sys_ptr->mesh_ptr_array.back()->number_of_nonplanar_blocks);

      glLineWidth(1.0f);
      glColor4f(1.0f, 1.0f, 0.0f, 0.7f);
      glEnableClientState(GL_VERTEX_ARRAY);

      glVertexPointer(3, GL_FLOAT, sizeof(My_Type::Vector3f),
                      this->render_engine.voxel_block_lines);
      glDrawArrays(GL_LINES, 0, this->render_engine.number_of_blocks * 24);

      glDisableClientState(GL_VERTEX_ARRAY);
    }
  }

  // 4. Draw current points cloud
  if (this->view_flag_list[4]) {
    // Transform to SLAM world coordinate
    this->OpenGL_draw_in_some_camera_coordinate(
        this->SLAM_system_ptr->estimated_camera_pose.mat);

    // Set property
    glColor4f(1.0f, 1.0f, 0.0f, 0.3f);

    glEnableClientState(GL_VERTEX_ARRAY);
    // glEnableClientState(GL_COLOR_ARRAY);

    const int layer_id = 0;
    glPointSize(1.0f + (float)layer_id);
    glPointSize(3.0f);
    glVertexPointer(3, GL_FLOAT, 0,
                    this->SLAM_system_ptr->preprocess_engine->hierarchy_points
                        .data_ptrs[layer_id]);
    // glColorPointer(3, GL_FLOAT, 0,
    // this->SLAM_system_ptr->preprocess_engine->hierarchy_normals.data_ptrs[0]);
    My_Type::Vector2i points_image_size =
        this->SLAM_system_ptr->preprocess_engine->hierarchy_points
            .size[layer_id];
    glDrawArrays(GL_POINTS, GLint(NULL),
                 points_image_size.width * points_image_size.height);

    glDisableClientState(GL_VERTEX_ARRAY);
    // glDisableClientState(GL_COLOR_ARRAY);
  }

  // 5. Draw model points cloud
  if (this->view_flag_list[5]) {
    if (false) {
      // Transform to SLAM world coordinate
      this->OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->pre_estimated_camera_pose.mat);

      // Set property
      glColor4f(1.0f, 0.0f, 1.0f, 0.7f);

      glEnableClientState(GL_VERTEX_ARRAY);
      // glEnableClientState(GL_COLOR_ARRAY);

      int layer_id = 0;
      glPointSize(1.0f + (float)layer_id);
      glPointSize(3.0f);
      glVertexPointer(3, GL_FLOAT, 0,
                      this->SLAM_system_ptr->preprocess_engine
                          ->hierarchy_model_points.data_ptrs[layer_id]);
      // glColorPointer(3, GL_FLOAT, 0,
      // this->SLAM_system_ptr->preprocess_engine->hierarchy_normals.data_ptrs[0]);
      My_Type::Vector2i points_image_size =
          this->SLAM_system_ptr->preprocess_engine->hierarchy_model_points
              .size[layer_id];
      glDrawArrays(GL_POINTS, GLint(NULL),
                   points_image_size.width * points_image_size.height);

      glDisableClientState(GL_VERTEX_ARRAY);
      // glDisableClientState(GL_COLOR_ARRAY);
    }

    if (true) {
      // Transform to SLAM world coordinate
      this->OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->pre_estimated_camera_pose.mat);

      // Set property
      glColor4f(0.0f, 1.0f, 1.0f, 0.7f);

      glEnableClientState(GL_VERTEX_ARRAY);

      int layer_id = 0;
      glPointSize(1.0f + (float)layer_id);
      glPointSize(3.0f);
      glVertexPointer(
          3, GL_FLOAT, 0,
          this->SLAM_system_ptr->track_engine->correspondence_lines);
      // glColorPointer(3, GL_FLOAT, 0,
      // this->SLAM_system_ptr->preprocess_engine->hierarchy_normals.data_ptrs[0]);
      My_Type::Vector2i points_image_size =
          this->SLAM_system_ptr->preprocess_engine->hierarchy_model_points
              .size[0];
      glDrawArrays(GL_LINES, GLint(NULL),
                   points_image_size.width * points_image_size.height * 2);

      glDisableClientState(GL_VERTEX_ARRAY);
    }

    //
    if (false) {
      this->render_engine.generate_normal_segment_line(
          this->SLAM_system_ptr->preprocess_engine->dev_hierarchy_model_points
              .data_ptrs[0],
          this->SLAM_system_ptr->preprocess_engine->dev_hierarchy_model_normals
              .data_ptrs[0],
          NormalsSource::MODEL_NORMAL);
      glColor4f(1.0f, 1.0f, 0.0f, 0.3f);
      //
      glLineWidth(1.0f);
      glEnableClientState(GL_VERTEX_ARRAY);
      glVertexPointer(
          3, GL_FLOAT, 0,
          this->render_engine.model_hierarchy_normal_to_draw.data_ptrs[0]);

      My_Type::Vector2i points_image_size =
          this->render_engine.model_hierarchy_normal_to_draw.size[0];
      glDrawArrays(GL_LINES, GLint(NULL),
                   points_image_size.width * points_image_size.height * 2);

      glDisableClientState(GL_VERTEX_ARRAY);
    }
  }

  // 6. Draw normals
  if (this->view_flag_list[6]) {
    // Model normal debug
    if (false) {
      this->render_engine.generate_normal_segment_line(
          this->SLAM_system_ptr->preprocess_engine->dev_raw_aligned_points,
          this->SLAM_system_ptr->preprocess_engine->dev_hierarchy_normals
              .data_ptrs[0],
          NormalsSource::DEPTH_NORMAL);

      // Transform to SLAM world coordinate
      this->OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->estimated_camera_pose.mat);

      // Set property
      glColor4f(1.0f, 0.0f, 1.0f, 0.3f);
      glLineWidth(1.0f);

      glEnableClientState(GL_VERTEX_ARRAY);

      glVertexPointer(
          3, GL_FLOAT, 0,
          this->render_engine.current_hierarchy_normal_to_draw.data_ptrs[0]);

      My_Type::Vector2i points_image_size =
          this->render_engine.current_hierarchy_normal_to_draw.size[0];
      glDrawArrays(GL_LINES, GLint(NULL),
                   points_image_size.width * points_image_size.height * 2);

      glDisableClientState(GL_VERTEX_ARRAY);
    }

    // Plane region mesh
    if (true) {
      //
      this->OpenGL_draw_in_OpenGL_camera_coordinate();

      // GLfloat light_color[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat light_color[] = {0.0, 0.0, 0.0, 1.0};
      glLightfv(GL_LIGHT0, GL_AMBIENT, light_color);
      GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 0.0};
      glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
      GLfloat light_specular[] = {0.0, 0.0, 0.0, 0.0};
      glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
      GLfloat light_position[] = {0.0, 0.0, 0.0, 1.0};
      // GLfloat light_position[] = { -this->GL_camera_Frame.mat[12],
      // this->GL_camera_Frame.mat[13], this->GL_camera_Frame.mat[14], 1.0 };
      glLightfv(GL_LIGHT0, GL_POSITION, light_position);

      // materials(GL_FRONT, &whiteLightMaterial);
      GLfloat material_ambient[] = {0.2, 0.2, 0.2, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient);
      GLfloat material_diffuse[] = {0.9, 0.9, 0.9, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse);
      GLfloat material_specular[] = {0.0, 0.0, 0.0, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular);
      GLfloat material_emmision[] = {0.0, 0.0, 0.0, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material_emmision);
      GLfloat material_shininess[] = {0.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess);

      this->OpenGL_draw_in_SLAM_world_coordinate();
      //
      glEnable(GL_LIGHTING);
      glEnable(GL_LIGHT0);

      //
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);

      glVertexPointer(
          3, GL_FLOAT, sizeof(My_Type::Vector3f),
          this->SLAM_system_ptr->mesh_of_total_map.planar_triangles);
      glNormalPointer(
          GL_FLOAT, 0,
          this->SLAM_system_ptr->mesh_of_total_map.planar_triangle_normals);

      // printf("number_of_planar_triangles = %d\n",
      // this->SLAM_system_ptr->mesh_of_total_map.number_of_planar_triangles);
      glDrawArrays(
          GL_TRIANGLES, GLint(NULL),
          this->SLAM_system_ptr->mesh_of_total_map.number_of_planar_triangles *
              3);
      // glDrawArrays(GL_POINTS, NULL,
      // this->SLAM_system_ptr->mesh_of_total_map.number_of_planar_triangles *
      // 3);

      glDisableClientState(GL_NORMAL_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);

      glDisable(GL_LIGHT0);
      glDisable(GL_LIGHTING);
    }

    // Render plane cell normal
    if (false) {
      this->OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->estimated_camera_pose.mat);

      //
      int cell_mat_size =
          this->SLAM_system_ptr->plane_detector->cell_mat_size.width *
          this->SLAM_system_ptr->plane_detector->cell_mat_size.height;

      //
      glPointSize(5.0f);
      glColor4f(1.0f, 0.0f, 1.0f, 0.7);
      glBegin(GL_POINTS);
      for (int cell_id = 0; cell_id < cell_mat_size; cell_id++) {
        Cell_info temp_cell =
            this->SLAM_system_ptr->plane_detector->cell_info_mat[cell_id];
        glVertex4f(temp_cell.x, temp_cell.y, temp_cell.z, 1.0f);
      }
      glEnd();

      glLineWidth(2.0f);
      glColor4f(1.0f, 0.0f, 0.0f, 0.7);
      glBegin(GL_LINES);
      for (int cell_id = 0; cell_id < cell_mat_size; cell_id++) {
        Cell_info temp_cell =
            this->SLAM_system_ptr->plane_detector->cell_info_mat[cell_id];
        glVertex4f(temp_cell.x, temp_cell.y, temp_cell.z, 1.0f);

        My_Type::Vector3f normal_inc(temp_cell.nx, temp_cell.ny, temp_cell.nz);
        normal_inc *= 0.05;
        glVertex4f(temp_cell.x + normal_inc.x, temp_cell.y + normal_inc.y,
                   temp_cell.z + normal_inc.z, 1.0f);
      }
      glEnd();
    }
  }

  // 7. Draw model surface
  if (this->view_flag_list[7]) {
    if (SLAM_system_settings::instance()->generate_mesh_for_visualization) {
      // Set Light at OpenGL camera viewpoint
      this->OpenGL_draw_in_OpenGL_camera_coordinate();
      // GLfloat light_color[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat light_color[] = {0.0, 0.0, 0.0, 1.0};
      glLightfv(GL_LIGHT0, GL_AMBIENT, light_color);
      GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 0.0};
      glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
      GLfloat light_specular[] = {0.0, 0.0, 0.0, 0.0};
      glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
      GLfloat light_position[] = {0.0, 0.0, 0.0, 1.0};
      glLightfv(GL_LIGHT0, GL_POSITION, light_position);

      // materialStruct whiteLightMaterial = {
      //	{ 0.0, 0.0, 0.0, 1.0 },// 环境反射系数
      //	{ 0.0, 0.0, 0.0, 1.0 },// 漫反射系数
      //	{ 0.0, 0.0, 0.0, 1.0 },// 镜面反射系数
      //	{ 1.0, 1.0, 1.0, 1.0 },// 发射光
      //	0	  // 镜面反射指数(反光指数)
      //};
      // materials(GL_FRONT, &whiteLightMaterial);
      GLfloat material_ambient[] = {0.2, 0.2, 0.2, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_ambient);
      GLfloat material_diffuse[] = {0.9, 0.9, 0.9, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_diffuse);
      GLfloat material_specular[] = {0.0, 0.0, 0.0, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular);
      GLfloat material_emmision[] = {0.0, 0.0, 0.0, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material_emmision);
      GLfloat material_shininess[] = {0.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess);

      //
      this->OpenGL_draw_in_SLAM_world_coordinate();
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      glPointSize(1.0f);
      glLineWidth(1.0f);
      /*glBegin(GL_LINES);
            for (int i = 0; i <
   this->SLAM_system_ptr->mesh_of_total_map.number_of_triangles; i++)
            {
                    My_Type::Vector3f p0 =
   this->SLAM_system_ptr->mesh_of_total_map.triangles[i * 3 + 0];
                    My_Type::Vector3f p1 =
   this->SLAM_system_ptr->mesh_of_total_map.triangles[i * 3 + 1];
                    My_Type::Vector3f p2 =
   this->SLAM_system_ptr->mesh_of_total_map.triangles[i * 3 + 2];
                    glVertex3f(p0.x, p0.y, p0.z);	glVertex3f(p1.x, p1.y,
   p1.z); glVertex3f(p1.x, p1.y, p1.z);	glVertex3f(p2.x, p2.y, p2.z);
                    glVertex3f(p2.x, p2.y, p2.z);	glVertex3f(p0.x, p0.y,
   p0.z);
            }
            glEnd();*/

      if (this->mesh_render_mode == 1) {
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glEnable(GL_COLOR_MATERIAL);
      } else {
        // glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
      }
      glEnable(GL_LIGHTING);
      glEnable(GL_LIGHT0);
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);
      if (this->mesh_render_mode == 1) glEnableClientState(GL_COLOR_ARRAY);

      glVertexPointer(3, GL_FLOAT, sizeof(My_Type::Vector3f),
                      this->SLAM_system_ptr->mesh_of_total_map.triangles);
      glNormalPointer(
          GL_FLOAT, 0,
          this->SLAM_system_ptr->mesh_of_total_map.triangle_normals);
      if (this->mesh_render_mode == 1)
        glColorPointer(4, GL_UNSIGNED_BYTE, 0,
                       this->SLAM_system_ptr->mesh_of_total_map.triangle_color);

      // glDrawArrays(GL_TRIANGLES, NULL,
      // this->SLAM_system_ptr->mesh_of_total_map.number_of_triangles * 3);
      glDrawArrays(
          GL_TRIANGLES, GLint(NULL),
          this->SLAM_system_ptr->mesh_of_total_map.number_of_triangles * 3);
      // glDrawArrays(GL_TRIANGLES, NULL, (SUBMAP_VOXEL_BLOCK_NUM *
      // VOXEL_BLOCK_SIZE / 2 * 3));

      // printf("%d\n", (int)SUBMAP_VOXEL_BLOCK_NUM * VOXEL_BLOCK_SIZE / 2 * 3);
      // printf("%f\n", SUBMAP_VOXEL_BLOCK_NUM * VOXEL_BLOCK_SIZE * 0.5 * 3 *
      // sizeof(float) / 1024.0f / 1024.0f); glDrawArrays(GL_POINTS, NULL,
      // this->SLAM_system_ptr->mesh_of_total_map.number_of_triangles * 3);

      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_NORMAL_ARRAY);
      if (this->mesh_render_mode == 1) glDisableClientState(GL_COLOR_ARRAY);

      glDisable(GL_LIGHT0);
      glDisable(GL_LIGHTING);
      glDisable(GL_COLOR_MATERIAL);

    } else {
      // Copy out current scene informations
      memcpy(this->render_engine.scene_points,
             this->SLAM_system_ptr->map_engine->scene_points,
             this->render_engine.scene_depth_size.width *
                 this->render_engine.scene_depth_size.height *
                 sizeof(My_Type::Vector3f));
      memcpy(this->render_engine.scene_normals,
             this->SLAM_system_ptr->map_engine->scene_normals,
             this->render_engine.scene_depth_size.width *
                 this->render_engine.scene_depth_size.height *
                 sizeof(My_Type::Vector3f));
      checkCudaErrors(cudaMemcpy(
          render_engine.dev_scene_plane_label,
          this->SLAM_system_ptr->map_engine->scene_plane_labels,
          this->render_engine.scene_depth_size.width *
              this->render_engine.scene_depth_size.height * sizeof(int),
          cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(
          render_engine.dev_scene_points_weight,
          this->SLAM_system_ptr->map_engine->scene_weight,
          this->render_engine.scene_depth_size.width *
              this->render_engine.scene_depth_size.height * sizeof(int),
          cudaMemcpyHostToDevice));
      // Render scene point
      this->render_engine.render_scene_points(this->main_viewport_render_mode);

      //
      this->OpenGL_draw_in_OpenGL_camera_coordinate();
      // this->OpenGL_draw_in_SLAM_world_coordinate(); /* Debug */

      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      glPointSize(1.0f);

      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);

      // glVertexPointer(3, GL_FLOAT, 0,
      // this->SLAM_system_ptr->map_engine->scene_points); glColorPointer(3,
      // GL_FLOAT, 0, this->SLAM_system_ptr->map_engine->scene_normals);
      glVertexPointer(3, GL_FLOAT, 0, this->render_engine.scene_points);
      glColorPointer(4, GL_UNSIGNED_BYTE, 0,
                     this->render_engine.scene_points_color);

      //      glDrawArrays(GL_POINTS, GLint(NULL), 640 * 480);
      int tmp_w, tmp_h;
      this->data_engine_ptr->get_color_image_size(tmp_w, tmp_h);
      glDrawArrays(GL_POINTS, GLint(NULL), tmp_w * tmp_h);

      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
    }
  }

  // 8. Draw current plane segmentation
  if (this->view_flag_list[8]) {
    // Show plane segmentation pseudo render result
    if (true) {
      // this->render_engine.pseudo_render_plane_labels(this->SLAM_system_ptr->plane_detector->dev_current_plane_labels);
      // this->render_engine.pseudo_render_plane_labels(this->SLAM_system_ptr->map_engine->dev_model_plane_labels);
      this->render_engine.pseudo_render_plane_labels(
          this->SLAM_system_ptr->plane_detector->dev_current_cell_labels);
      // Submap_SLAM_system * sys_ptr = dynamic_cast<Submap_SLAM_system
      // *>(this->SLAM_system_ptr); Submap_Voxel_map * map_ptr =
      // dynamic_cast<Submap_Voxel_map *>(sys_ptr->submap_ptr_array.back());
      // this->render_engine.pseudo_render_plane_labels(map_ptr->dev_model_plane_labels);

      // Transform to SLAM camera coordinate
      this->OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->estimated_camera_pose.mat);

      // Set property
      glColor4f(1.0f, 1.0f, 1.0f, 0.5f);
      glLineWidth(1.0f);

      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);

      glVertexPointer(3, GL_FLOAT, 0,
                      this->SLAM_system_ptr->preprocess_engine->hierarchy_points
                          .data_ptrs[0]);
      glColorPointer(4, GL_UNSIGNED_BYTE, 0,
                     this->render_engine.pseudo_plane_color);

      My_Type::Vector2i points_image_size =
          this->SLAM_system_ptr->preprocess_engine->hierarchy_points.size[0];
      glDrawArrays(GL_POINTS, GLint(NULL),
                   points_image_size.width * points_image_size.height);

      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);

      //
      this->OpenGL_draw_in_SLAM_world_coordinate();
      // Draw plane vectors
      if (false) {
        glBegin(GL_LINES);
        glLineWidth(5.0);
        glColor4f(0.0f, 1.0f, 1.0f, 0.5f);
        for (int plane_id = 0;
             plane_id <
             this->SLAM_system_ptr->plane_detector->current_plane_counter;
             plane_id++) {
          if (!this->SLAM_system_ptr->plane_detector->current_planes[plane_id]
                   .is_valid)
            continue;

          My_Type::Vector3f plane_vector;
          plane_vector.x =
              -this->SLAM_system_ptr->plane_detector->current_planes[plane_id]
                   .nx;
          plane_vector.y =
              -this->SLAM_system_ptr->plane_detector->current_planes[plane_id]
                   .ny;
          plane_vector.z =
              -this->SLAM_system_ptr->plane_detector->current_planes[plane_id]
                   .nz;
          plane_vector *=
              this->SLAM_system_ptr->plane_detector->current_planes[plane_id].d;

          glVertex3f(0, 0, 0);
          glVertex3f(plane_vector.x, plane_vector.y, plane_vector.z);
        }
        glEnd();
      }
    }

    // Draw model plane coordiante
    if (false) {
      Basic_Voxel_map *map_ptr =
          dynamic_cast<Basic_Voxel_map *>(this->SLAM_system_ptr->map_engine);

      Eigen::Matrix4f coordinate_pose_mat;
      for (int plane_id = 1; plane_id < map_ptr->plane_map_ptr->plane_counter;
           plane_id++) {
        //
        coordinate_pose_mat.setIdentity();
        coordinate_pose_mat(0, 0) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.x;
        coordinate_pose_mat(0, 1) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.y;
        coordinate_pose_mat(0, 2) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.z;
        coordinate_pose_mat(1, 0) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.x;
        coordinate_pose_mat(1, 1) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.y;
        coordinate_pose_mat(1, 2) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.z;
        coordinate_pose_mat(2, 0) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.x;
        coordinate_pose_mat(2, 1) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.y;
        coordinate_pose_mat(2, 2) =
            map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.z;
        coordinate_pose_mat.block(0, 0, 3, 3).transposeInPlace();
        float distance = map_ptr->plane_map_ptr->plane_list[plane_id].d;
        coordinate_pose_mat(0, 3) =
            -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.x *
            distance;
        coordinate_pose_mat(1, 3) =
            -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.y *
            distance;
        coordinate_pose_mat(2, 3) =
            -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.z *
            distance;
        //
        this->OpenGL_draw_in_SLAM_world_coordinate();
        glMultMatrixf(coordinate_pose_mat.data());

        //
        draw_coordinate_GL(0.1, 1);
      }

      static bool debug_bool = false;
      if (!debug_bool) {
        debug_bool = true;
        for (int plane_id = 1; plane_id < map_ptr->plane_map_ptr->plane_counter;
             plane_id++) {
          //
          coordinate_pose_mat.setIdentity();
          coordinate_pose_mat(0, 0) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.x;
          coordinate_pose_mat(0, 1) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.y;
          coordinate_pose_mat(0, 2) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].x_vec.z;
          coordinate_pose_mat(1, 0) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.x;
          coordinate_pose_mat(1, 1) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.y;
          coordinate_pose_mat(1, 2) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].y_vec.z;
          coordinate_pose_mat(2, 0) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.x;
          coordinate_pose_mat(2, 1) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.y;
          coordinate_pose_mat(2, 2) =
              map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.z;
          coordinate_pose_mat.block(0, 0, 3, 3).transposeInPlace();
          float distance = map_ptr->plane_map_ptr->plane_list[plane_id].d;
          coordinate_pose_mat(0, 3) =
              -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.x *
              distance;
          coordinate_pose_mat(1, 3) =
              -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.y *
              distance;
          coordinate_pose_mat(2, 3) =
              -map_ptr->plane_map_ptr->plane_coordinate_list[plane_id].z_vec.z *
              distance;
          cout << coordinate_pose_mat << endl << endl;
        }
      }
    }
  }

  //
  if (this->view_flag_list[9]) {
    if (false) {
      this->OpenGL_draw_in_SLAM_world_coordinate();

      glColor4f(0.0f, 1.0f, 1.0f, 0.7f);
      glLineWidth(2.0f);
      glEnableClientState(GL_VERTEX_ARRAY);

      Basic_Voxel_map *map_ptr =
          dynamic_cast<Basic_Voxel_map *>(this->SLAM_system_ptr->map_engine);
      glVertexPointer(3, GL_FLOAT, 0, map_ptr->plane_map_ptr->block_vertexs);
      glDrawArrays(GL_LINES, GLint(NULL),
                   map_ptr->plane_map_ptr->number_of_pixel_blocks * 8);
      // glDrawArrays(GL_POINTS, NULL,
      // map_ptr->plane_map_ptr->number_of_pixel_blocks * 8);

      glDisableClientState(GL_VERTEX_ARRAY);

      static bool temp_b = true;
      if (temp_b) {
        temp_b = false;
        for (size_t i = 0; i < 1000; i++) {
          printf("%f, %f, %f\n", map_ptr->plane_map_ptr->block_vertexs[i].x,
                 map_ptr->plane_map_ptr->block_vertexs[i].y,
                 map_ptr->plane_map_ptr->block_vertexs[i].z);
        }
      }
    }

    if (true) {
      this->OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *sys_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      Submap_Voxel_map *map_ptr =
          dynamic_cast<Submap_Voxel_map *>(sys_ptr->submap_ptr_array.back());
      //

      glColor4f(0.0f, 0.7f, 1.0f, 0.7f);
      glLineWidth(2.0f);
      glEnableClientState(GL_VERTEX_ARRAY);

      glVertexPointer(3, GL_FLOAT, 0, map_ptr->plane_map_ptr->block_vertexs);
      glDrawArrays(GL_LINES, GLint(NULL),
                   map_ptr->plane_map_ptr->number_of_pixel_blocks * 8);
      // glDrawArrays(GL_POINTS, NULL,
      // map_ptr->plane_map_ptr->number_of_pixel_blocks * 8);

      glDisableClientState(GL_VERTEX_ARRAY);
    }
  }

  // 10. Model features
  if (this->view_flag_list[0]) {
    float half_cube_size = (float)(VOXEL_SIZE * 0.2 * 8);

    // Draw keypoint in each submap
    if (true) {
      OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      // slam_system_ptr->
      glColor4f(0.1f, 0.9f, 0.1f, 0.9f);
      glLineWidth(4.0f);
      int number_of_submaps = slam_system_ptr->feature_map_ptr_array.size();
      for (int map_id = 0; map_id < number_of_submaps; map_id++) {
        Eigen::Matrix4f map_pose =
            slam_system_ptr->submap_pose_array[map_id]->mat;

        int number_of_keypoints = slam_system_ptr->feature_map_ptr_array[map_id]
                                      ->model_keypoints.size();
        for (int point_id = 0; point_id < number_of_keypoints; point_id++) {
          My_Type::Vector3f keypoint_vec =
              slam_system_ptr->feature_map_ptr_array[map_id]
                  ->model_keypoints[point_id]
                  .point;
          if (!slam_system_ptr->feature_map_ptr_array[map_id]
                   ->model_keypoints[point_id]
                   .is_valid)
            continue;

          Eigen::Vector4f keypoint_vec_eigen(keypoint_vec.x, keypoint_vec.y,
                                             keypoint_vec.z, 1.0f);
          keypoint_vec_eigen = map_pose * keypoint_vec_eigen.eval();

          glTranslatef(keypoint_vec_eigen.x(), keypoint_vec_eigen.y(),
                       keypoint_vec_eigen.z());
          glutWireCube(half_cube_size * 2);
          glTranslatef(-keypoint_vec_eigen.x(), -keypoint_vec_eigen.y(),
                       -keypoint_vec_eigen.z());
        }
      }
    }

    // Draw keypoint association between submaps
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      glColor4f(1.0f, 0.0f, 0.0f, 0.7f);
      glLineWidth(3.0f);
      glBegin(GL_LINES);
      for (int match_id = 0;
           match_id <
           slam_system_ptr->keypoint_associator->all_submap_id_pair.size();
           match_id++) {
        int map_1 =
            slam_system_ptr->keypoint_associator->all_submap_id_pair[match_id]
                .first;
        int map_2 =
            slam_system_ptr->keypoint_associator->all_submap_id_pair[match_id]
                .second;

        Eigen::Matrix4f map_1_pose =
            slam_system_ptr->submap_pose_array[map_1]->mat;
        Eigen::Matrix4f map_2_pose =
            slam_system_ptr->submap_pose_array[map_2]->mat;

        for (int point_pair_id = 0;
             point_pair_id <
             slam_system_ptr->keypoint_associator->all_matches[match_id].size();
             point_pair_id++) {
          int map_1_point_id = slam_system_ptr->keypoint_associator
                                   ->all_matches[match_id][point_pair_id]
                                   .first;
          int map_2_point_id = slam_system_ptr->keypoint_associator
                                   ->all_matches[match_id][point_pair_id]
                                   .second;

          My_Type::Vector3f keypoint_1_vec =
              slam_system_ptr->feature_map_ptr_array[map_1]
                  ->model_keypoints[map_1_point_id]
                  .point;
          My_Type::Vector3f keypoint_2_vec =
              slam_system_ptr->feature_map_ptr_array[map_2]
                  ->model_keypoints[map_2_point_id]
                  .point;

          Eigen::Vector4f keypoint_1_eigen(keypoint_1_vec.x, keypoint_1_vec.y,
                                           keypoint_1_vec.z, 1.0f);
          keypoint_1_eigen = map_1_pose * keypoint_1_eigen.eval();
          Eigen::Vector4f keypoint_2_eigen(keypoint_2_vec.x, keypoint_2_vec.y,
                                           keypoint_2_vec.z, 1.0f);
          keypoint_2_eigen = map_2_pose * keypoint_2_eigen.eval();

          glVertex3f(keypoint_1_eigen.x(), keypoint_1_eigen.y(),
                     keypoint_1_eigen.z());
          glVertex3f(keypoint_2_eigen.x(), keypoint_2_eigen.y(),
                     keypoint_2_eigen.z());
        }
      }
      glEnd();
    }

    // Draw keyframe's keypoint weight center
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      int number_of_submaps = slam_system_ptr->feature_map_ptr_array.size();
      for (int map_id = 0; map_id < number_of_submaps; map_id++) {
        Eigen::Matrix4f map_pose =
            slam_system_ptr->submap_pose_array[map_id]->mat;

        int number_of_keyframe = slam_system_ptr->feature_map_ptr_array[map_id]
                                     ->keyframe_weigth_centers.size();
        for (int point_id = 0; point_id < number_of_keyframe; point_id++) {
          My_Type::Vector3f weight_center =
              slam_system_ptr->feature_map_ptr_array[map_id]
                  ->keyframe_weigth_centers[point_id];

          Eigen::Vector4f keypoint_vec_eigen(weight_center.x, weight_center.y,
                                             weight_center.z, 1.0f);
          keypoint_vec_eigen = map_pose * keypoint_vec_eigen.eval();

          glTranslatef(keypoint_vec_eigen.x(), keypoint_vec_eigen.y(),
                       keypoint_vec_eigen.z());
          glutSolidSphere(0.02 * UI_parameters::instance()->GL_view_scale, 16,
                          16);
          glTranslatef(-keypoint_vec_eigen.x(), -keypoint_vec_eigen.y(),
                       -keypoint_vec_eigen.z());
        }
      }
    }

    // Draw looped keypoints
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      glColor4f(0.6f, 1.0f, 0.6f, 0.7f);
      glLineWidth(1.0f);
      for (int i = 0; i < slam_system_ptr->keypoint_buffer_1.size(); i++) {
        glTranslatef(slam_system_ptr->keypoint_buffer_1[i].x(),
                     slam_system_ptr->keypoint_buffer_1[i].y(),
                     slam_system_ptr->keypoint_buffer_1[i].z());
        glutWireCube(half_cube_size);
        glTranslatef(-slam_system_ptr->keypoint_buffer_1[i].x(),
                     -slam_system_ptr->keypoint_buffer_1[i].y(),
                     -slam_system_ptr->keypoint_buffer_1[i].z());
      }
      glColor4f(1.0f, 1.0f, 0.0, 0.7f);
      glLineWidth(1.0f);
      for (int i = 0; i < slam_system_ptr->keypoint_buffer_2.size(); i++) {
        glTranslatef(slam_system_ptr->keypoint_buffer_2[i].x(),
                     slam_system_ptr->keypoint_buffer_2[i].y(),
                     slam_system_ptr->keypoint_buffer_2[i].z());
        glutWireCube(half_cube_size);
        glTranslatef(-slam_system_ptr->keypoint_buffer_2[i].x(),
                     -slam_system_ptr->keypoint_buffer_2[i].y(),
                     -slam_system_ptr->keypoint_buffer_2[i].z());
      }

      // Association
      // glColor4f(1.0f, 0.0f, 0.0f, 0.7f);
      // glLineWidth(5.0f);
      // glBegin(GL_LINES);
      // for (int i = 0; i < slam_system_ptr->keypoint_buffer_2.size(); i++)
      //{
      //	glVertex3f(slam_system_ptr->keypoint_buffer_1[i].x(),
      // slam_system_ptr->keypoint_buffer_1[i].y(),
      // slam_system_ptr->keypoint_buffer_1[i].z());
      //	glVertex3f(slam_system_ptr->keypoint_buffer_2[i].x(),
      // slam_system_ptr->keypoint_buffer_2[i].y(),
      // slam_system_ptr->keypoint_buffer_2[i].z());
      //}
      // glEnd();

      // Loop frame keypoints
    }

    // Draw keypoint hash block
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);
      glColor4f(1.0f, 1.0f, 1.0f, 0.7f);
      glLineWidth(1.0f);
      int number_of_submaps = slam_system_ptr->feature_map_ptr_array.size();
      for (int map_id = 0; map_id < number_of_submaps; map_id++) {
        for (std::unordered_map<My_Type::Vector3i, int>::iterator kp_block_it =
                 slam_system_ptr->feature_map_ptr_array.back()
                     ->map_point_mapper.begin();
             kp_block_it != slam_system_ptr->feature_map_ptr_array.back()
                                ->map_point_mapper.end();
             ++kp_block_it) {
          My_Type::Vector3i block_veci = kp_block_it->first;
          My_Type::Vector3f block_vecf(block_veci.x, block_veci.y,
                                       block_veci.z);
          block_vecf = block_vecf * FEATRUE_BLOCK_WIDTH;
          // block_vecf += FEATRUE_BLOCK_WIDTH * 0.5;

          glTranslatef(block_vecf.x, block_vecf.y, block_vecf.z);
          glutWireCube((FEATRUE_BLOCK_WIDTH));
          glTranslatef(-block_vecf.x, -block_vecf.y, -block_vecf.z);
        }
      }
    }

    // Draw keypoint in current frame
    if (false) {
      OpenGL_draw_in_some_camera_coordinate(
          this->SLAM_system_ptr->estimated_camera_pose.mat);

      glColor4f(0.6f, 0.1f, 0.6f, 0.7f);
      glLineWidth(1.0f);
      int number_of_current_keypoints = this->SLAM_system_ptr->feature_detector
                                            ->current_keypoint_position.size();
      for (int keypoint_id = 0; keypoint_id < number_of_current_keypoints;
           keypoint_id++) {
        My_Type::Vector3f keypoint_vec =
            this->SLAM_system_ptr->feature_detector
                ->current_keypoint_position[keypoint_id];
        if (this->SLAM_system_ptr->feature_detector
                ->current_match_to_model_id[keypoint_id] < 0)
          continue;
        if (keypoint_vec.z < FLT_EPSILON) continue;

        glTranslatef(keypoint_vec.x, keypoint_vec.y, keypoint_vec.z);
        glutWireCube(half_cube_size * 1.5f);
        glTranslatef(-keypoint_vec.x, -keypoint_vec.y, -keypoint_vec.z);
      }
      glColor4f(0.6f, 0.6f, 0.1f, 0.7f);
      glLineWidth(1.0f);
      for (int keypoint_id = 0; keypoint_id < number_of_current_keypoints;
           keypoint_id++) {
        My_Type::Vector3f keypoint_vec =
            this->SLAM_system_ptr->feature_detector
                ->current_keypoint_position[keypoint_id];
        if (this->SLAM_system_ptr->feature_detector
                ->current_match_to_model_id[keypoint_id] >= 0)
          continue;
        if (keypoint_vec.z < FLT_EPSILON) continue;

        glTranslatef(keypoint_vec.x, keypoint_vec.y, keypoint_vec.z);
        glutWireCube(half_cube_size * 1.5f);
        glTranslatef(-keypoint_vec.x, -keypoint_vec.y, -keypoint_vec.z);
      }
    }

    // Draw neighbor keypoint
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();

      glColor4f(0.0f, 0.0f, 1.0f, 0.7f);
      glLineWidth(1.0f);
      int number_of_current_keypoints = this->SLAM_system_ptr->feature_detector
                                            ->visible_model_keypoints.size();
      for (int keypoint_id = 0; keypoint_id < number_of_current_keypoints;
           keypoint_id++) {
        My_Type::Vector3f check_vec =
            this->SLAM_system_ptr->feature_detector
                ->current_keypoint_position[keypoint_id];
        if (check_vec.z < FLT_EPSILON) continue;

        for (int neighbor_id = 0;
             neighbor_id < this->SLAM_system_ptr->feature_detector
                               ->visible_model_keypoints[keypoint_id]
                               .size();
             neighbor_id++) {
          My_Type::Vector3f keypoint_vec =
              this->SLAM_system_ptr->feature_detector
                  ->visible_model_keypoints[keypoint_id][neighbor_id];
          glTranslatef(keypoint_vec.x, keypoint_vec.y, keypoint_vec.z);
          glutWireCube(half_cube_size * 0.5f);
          glTranslatef(-keypoint_vec.x, -keypoint_vec.y, -keypoint_vec.z);
        }
      }
    }

    // Draw current-to-model keypoint matches
    if (false) {
      OpenGL_draw_in_SLAM_world_coordinate();
      Eigen::Matrix4f camera_pose =
          this->SLAM_system_ptr->estimated_camera_pose.mat;

      Submap_SLAM_system *slam_system_ptr =
          dynamic_cast<Submap_SLAM_system *>(this->SLAM_system_ptr);

      glColor4f(1.0f, 0.1f, 0.6f, 0.7f);
      glLineWidth(2.0f);
      int number_of_current_keypoints = this->SLAM_system_ptr->feature_detector
                                            ->current_keypoint_position.size();
      glBegin(GL_LINES);
      for (int current_id = 0; current_id < number_of_current_keypoints;
           current_id++) {
        int model_kp_id = this->SLAM_system_ptr->feature_detector
                              ->current_match_to_model_id[current_id];
        if (model_kp_id < 0) continue;

        My_Type::Vector3f current_kp1 =
            this->SLAM_system_ptr->feature_detector
                ->current_keypoint_position[current_id];
        Eigen::Vector3f current_kp(current_kp1.x, current_kp1.y, current_kp1.z);
        current_kp = camera_pose.block(0, 0, 3, 3) * current_kp.eval() +
                     camera_pose.block(0, 3, 3, 1);
        glVertex3f(current_kp.x(), current_kp.y(), current_kp.z());

        My_Type::Vector3f model_kp =
            slam_system_ptr->feature_map_ptr_array.back()
                ->model_keypoints[model_kp_id]
                .point;
        glVertex3f(model_kp.x, model_kp.y, model_kp.z);
      }
      glEnd();

      // glColor4f(0.0f, 0.4f, 1.0f, 0.7f);
      // glPointSize(4.0f);
      // glBegin(GL_POINTS);
      // for (int current_id = 0; current_id < number_of_current_keypoints;
      // current_id++)
      //{
      //	int model_kp_id =
      // this->SLAM_system_ptr->feature_detector->current_match_to_model_id[current_id];
      //	if (model_kp_id < 0)	continue;
      //	My_Type::Vector3f current_kp1 =
      // this->SLAM_system_ptr->feature_detector->current_keypoint_position[current_id];
      //	Eigen::Vector3f current_kp(current_kp1.x, current_kp1.y,
      // current_kp1.z); 	current_kp = camera_pose.block(0, 0, 3, 3) *
      // current_kp.eval() + camera_pose.block(0, 3, 3, 1);
      //	glVertex3f(current_kp.x(), current_kp.y(), current_kp.z());
      //	My_Type::Vector3f model_kp =
      // slam_system_ptr->feature_map_ptr_array.back()->model_keypoints[model_kp_id].point;
      //	glVertex3f(model_kp.x, model_kp.y, model_kp.z);
      //}
      // glEnd();
    }
  }

  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  if (false) {
    // Render sub-viewport-2
    switch (this->sub_viewport2_mode) {
      case SubViewport2Mode::PSEUDO_DEPTH_IMAGE: {
        this->render_engine.pseudo_render_depth(
            this->SLAM_system_ptr->preprocess_engine->dev_raw_aligned_points);
        break;
      }
      case SubViewport2Mode::NORMAL_IMAGE: {
        break;
      }
      case SubViewport2Mode::RESIDUAL_IMAGE: {
        break;
      }
      default:
        break;
    }

    //
    glViewport(0, 0, this->main_viewport_width, this->main_viewport_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw current frame
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->sub1_viewport_texture);
    //
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    this->render_engine.depth_size.width,
                    this->render_engine.depth_size.height, GL_RGBA,
                    GL_UNSIGNED_BYTE, this->render_engine.viewport_2_color);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, +1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
  }
}

void UI_engine::render_sub_viewport1() {
  if (this->SLAM_system_ptr->color_mat.empty()) return;

  if (true) {
    drawKeypoints(this->SLAM_system_ptr->color_mat.clone(),
                  this->SLAM_system_ptr->feature_detector->current_keypoints,
                  this->SLAM_system_ptr->color_mat, cv::Scalar(0, 0, 255),
                  cv::DrawMatchesFlags::DEFAULT);
  }

  glViewport(this->main_viewport_width, this->sub_viewport_height,
             this->sub_viewport_width, this->sub_viewport_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Draw current frame
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, this->sub1_viewport_texture);
  //
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->SLAM_system_ptr->color_mat.cols,
                  this->SLAM_system_ptr->color_mat.rows, GL_BGR,
                  GL_UNSIGNED_BYTE, this->SLAM_system_ptr->color_mat.data);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(+1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(+1.0f, +1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, +1.0f);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void UI_engine::render_sub_viewport2() {
  switch (this->sub_viewport2_mode) {
    case SubViewport2Mode::PSEUDO_DEPTH_IMAGE: {
      this->render_engine.pseudo_render_depth(
          this->SLAM_system_ptr->preprocess_engine->dev_raw_aligned_points);
      break;
    }
    case SubViewport2Mode::NORMAL_IMAGE: {
      break;
    }
    case SubViewport2Mode::RESIDUAL_IMAGE: {
      break;
    }
    case SubViewport2Mode::SVP2MODE_NUM: {
      break;
    }
    default:
      break;
  }

  glViewport(this->main_viewport_width, 0, this->sub_viewport_width,
             this->sub_viewport_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Draw current frame
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, this->sub1_viewport_texture);

  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->render_engine.depth_size.width,
                  this->render_engine.depth_size.height, GL_RGBA,
                  GL_UNSIGNED_BYTE, this->render_engine.viewport_2_color);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(+1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(+1.0f, +1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, +1.0f);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

#if __unix__
#pragma region "OpenGL draw coordinate transformation" {
#elif _WIN32
#pragma region(OpenGL draw coordinate transformation)
#endif

void UI_engine::OpenGL_draw_in_OpenGL_camera_coordinate() {
  glMatrixMode(GL_MODELVIEW);
  // Transform to current OpenGL view camera coordinate (current screen)
  glLoadIdentity();
  glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
}

void UI_engine::OpenGL_draw_in_OpenGL_world_coordinate() {
  glMatrixMode(GL_MODELVIEW);
  // Transform to OpenGL world coordinate
  my_load_frame(this->GL_camera_Frame);
}

void UI_engine::OpenGL_draw_in_SLAM_world_coordinate() {
  glMatrixMode(GL_MODELVIEW);
  // Transform to OpenGL world coordinate
  my_load_frame(this->GL_camera_Frame);
  // Transform to SLAM world coordinate
  glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
}

void UI_engine::OpenGL_draw_in_some_camera_coordinate(
    Eigen::Matrix4f camera_pose) {
  glMatrixMode(GL_MODELVIEW);
  // Transform to OpenGL world coordinate
  my_load_frame(this->GL_camera_Frame);
  // Transform to SLAM world coordinate
  glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
  // Transform to SLAM camera coordinate
  camera_pose.block(0, 3, 3, 1) *= UI_parameters::instance()->GL_view_scale;
  glMultMatrixf(camera_pose.data());
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "OpenGL user interactive events" {
#elif _WIN32
#pragma region(OpenGL user interactive events)
#endif

void UI_engine::OpenGL_NormalKeyFunction(unsigned char key, int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // Record keypress state
  UI_ptr->normal_key[key] = 1;
  // Show object of reference
  UI_ptr->show_reference_object = true;

  if (UI_ptr->normal_key[uchar('1')] == 1 ||
      UI_ptr->normal_key[uchar('!')] == 1)
    UI_ptr->view_flag_list[1] = !UI_ptr->view_flag_list[1];
  if (UI_ptr->normal_key[uchar('2')] == 1 ||
      UI_ptr->normal_key[uchar('@')] == 1)
    UI_ptr->view_flag_list[2] = !UI_ptr->view_flag_list[2];
  if (UI_ptr->normal_key[uchar('3')] == 1 ||
      UI_ptr->normal_key[uchar('#')] == 1)
    UI_ptr->view_flag_list[3] = !UI_ptr->view_flag_list[3];
  if (UI_ptr->normal_key[uchar('4')] == 1 ||
      UI_ptr->normal_key[uchar('$')] == 1)
    UI_ptr->view_flag_list[4] = !UI_ptr->view_flag_list[4];
  if (UI_ptr->normal_key[uchar('5')] == 1 ||
      UI_ptr->normal_key[uchar('%')] == 1)
    UI_ptr->view_flag_list[5] = !UI_ptr->view_flag_list[5];
  if (UI_ptr->normal_key[uchar('6')] == 1 ||
      UI_ptr->normal_key[uchar('^')] == 1)
    UI_ptr->view_flag_list[6] = !UI_ptr->view_flag_list[6];
  if (UI_ptr->normal_key[uchar('7')] == 1 ||
      UI_ptr->normal_key[uchar('&')] == 1)
    UI_ptr->view_flag_list[7] = !UI_ptr->view_flag_list[7];
  if (UI_ptr->normal_key[uchar('8')] == 1 ||
      UI_ptr->normal_key[uchar('*')] == 1)
    UI_ptr->view_flag_list[8] = !UI_ptr->view_flag_list[8];
  if (UI_ptr->normal_key[uchar('9')] == 1 ||
      UI_ptr->normal_key[uchar('(')] == 1)
    UI_ptr->view_flag_list[9] = !UI_ptr->view_flag_list[9];
  if (UI_ptr->normal_key[uchar('0')] == 1 ||
      UI_ptr->normal_key[uchar(')')] == 1)
    UI_ptr->view_flag_list[0] = !UI_ptr->view_flag_list[0];

  // Sub viewport2 render mode
  if (UI_ptr->normal_key[uchar('`')] == 1 ||
      UI_ptr->normal_key[uchar('~')] == 1) {
    UI_ptr->sub_viewport2_mode =
        (SubViewport2Mode)((int)UI_ptr->sub_viewport2_mode + 1);
    if (UI_ptr->sub_viewport2_mode >= SubViewport2Mode::SVP2MODE_NUM)
      UI_ptr->sub_viewport2_mode = SubViewport2Mode::PSEUDO_DEPTH_IMAGE;
  }
  // Main viewport render mode
  if (UI_ptr->normal_key[uchar('r')] == 1 ||
      UI_ptr->normal_key[uchar('R')] == 1) {
    UI_ptr->main_viewport_render_mode =
        (MainViewportRenderMode)((int)UI_ptr->main_viewport_render_mode + 1);
    if (UI_ptr->main_viewport_render_mode >=
        MainViewportRenderMode::MVPRMODE_NUM)
      UI_ptr->main_viewport_render_mode = MainViewportRenderMode::PHONG_RENDER;
  }

  // Do nothing.
  if (UI_ptr->normal_key[uchar('t')] == 1 ||
      UI_ptr->normal_key[uchar('T')] == 1) {
    // if (UI_ptr->render_object_id <
    // main_engine_ptr->map_engine_ptr->fragment_index + 1)
    //{
    //	UI_ptr->render_object_id++;
    //}
    // else
    //{
    //	UI_ptr->render_object_id = 0;
    //}
  }

  // Do nothing.
  if (UI_ptr->normal_key[uchar('g')] == 1 ||
      UI_ptr->normal_key[uchar('G')] == 1) {
    //// match list
    // if (UI_ptr->render_match_list_id <
    // main_engine_ptr->map_engine_ptr->feature_map_match_graph.size() - 1)
    //{
    //	UI_ptr->render_match_list_id++;
    //}
    // else
    //{
    //	UI_ptr->render_match_list_id = 0;
    //}
  }

  // Print info for DGBUG.
  if ('p' == key || 'P' == key) {
    //    std::cout << "GL camera transfer matrix is :" << std::endl;
    // UI_ptr->GL_camera_Frame.print();
    // cout << main_engine_ptr->pose_estimate.mat << endl;
  }

  // Not sure.
  if (';' == key || ':' == key) {
    Eigen::Matrix4f temp_pose;
    static int view_pose_id = 0;
    if (view_pose_id == 0) {
      temp_pose << +1.000, +0.000, +0.000, +0.000, +0.000, +1.000, +0.000,
          +0.000, +0.000, +0.000, +1.000, +0.000, -0.890, -0.065, +1.650,
          +1.000;
      view_pose_id++;
    } else if (view_pose_id == 1) {
      temp_pose << +0.796, +0.167, -0.582, +0.000, +0.000, +0.961, +0.276,
          +0.000, +0.605, -0.220, +0.765, +0.000, +1.677, -1.336, +1.967,
          +1.000;
      view_pose_id = 0;
    }

    float *ptr_f1 = UI_ptr->GL_camera_Frame.mat.data;
    float *ptr_f2 = temp_pose.data();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ptr_f1[i * 4 + j] = ptr_f2[j * 4 + i];
      }
    }
  }

  // switch estimate camera pose BUGS
  if ('o' == key || 'O' == key) {
    /*std::cout << "GL camera extrinsic matrix inverse is :" << std::endl;
        UI_ptr->GL_camera_Frame.print();*/

    Eigen::Matrix4f current_pose_mat =
        UI_ptr->SLAM_system_ptr->estimated_camera_pose.mat.eval();

    // Coordinate transformation
    Eigen::Matrix4f coordinate_change;
    coordinate_change.setIdentity();
    // coordinate_change(0, 0) = -1;
    coordinate_change(1, 1) = -1;
    coordinate_change(2, 2) = -1;

    //
    current_pose_mat.block(0, 0, 3, 3) =
        coordinate_change * current_pose_mat.block(0, 0, 3, 3).eval() *
        coordinate_change.inverse();
    current_pose_mat.block(0, 3, 3, 1) =
        coordinate_change * current_pose_mat.block(0, 3, 3, 1).eval();
    current_pose_mat = current_pose_mat.inverse().eval();

    float *ptr_f1 = UI_ptr->GL_camera_Frame.mat.data;
    //

    float *ptr_f2 = current_pose_mat.data();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ptr_f1[i * 4 + j] = ptr_f2[i * 4 + j];
      }
    }
  }

  // Force to stop WITH BUGS
  if ('i' == key || 'I' == key) {
    UI_ptr->SLAM_system_ptr->all_data_process_done = true;
    UI_ptr->SLAM_system_ptr->end_of_process_data();
  }

  // Screen shot one frame
  if ('l' == key || 'L' == key) {
    // Screen shot counter
    static int screen_shot_counter = 0;
    string screen_shot_path = "main_viewport_screenshot";
    screen_shot(screen_shot_path, screen_shot_counter);

    // to next shot counter
    screen_shot_counter++;
  }

  // Screen shot contious frame from current frame.
  if ('k' == key || 'K' == key) {
    UI_ptr->record_continuous = !UI_ptr->record_continuous;
  }

  // Mesh render mode, draw plane with color.
  if ('r' == key || 'R' == key) {
    UI_ptr->mesh_render_mode++;
    if (UI_ptr->mesh_render_mode >= 3) UI_ptr->mesh_render_mode = 0;
  }

  // GLCamera fov--
  if ('[' == key || '{' == key) {
    if (UI_parameters::instance()->GL_camera_fov >= 10.0f)
      UI_parameters::instance()->GL_camera_fov -= 5.0f;
  }

  // GLCamera fov++
  if (']' == key || '}' == key) {
    if (UI_parameters::instance()->GL_camera_fov >= 10.0f)
      UI_parameters::instance()->GL_camera_fov += 5.0f;
  }

  // Key Esc, exit.
  if (UI_ptr->normal_key[27] == 1) {
#ifdef LOGGING
    LOG_INFO("Exit by key Esc pressed event");
#endif
    exit(0);
  }
}

void UI_engine::OpenGL_NormalKeyUpFunction(unsigned char key, int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // Update keypress state
  UI_ptr->normal_key[key] = 0;

  // Hide object of reference
  UI_ptr->show_reference_object = false;
}

void UI_engine::OpenGL_SpecialKeyFunction(int key, int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // Update keypress state
  UI_ptr->special_key[key] = 1;
  // Show object of reference
  UI_ptr->show_reference_object = true;
}

void UI_engine::OpenGL_SpecialKeyUpFunction(int key, int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // Update keypress state
  UI_ptr->special_key[key] = 0;
  // Hide object of reference
  UI_ptr->show_reference_object = false;
}

void UI_engine::OpenGL_MouseButtonFunction(int button, int state, int x,
                                           int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // Update mouse position
  UI_ptr->mouse_X = x;
  UI_ptr->mouse_Y = y;

  // Left button pressed
  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN) &&
      (!UI_ptr->mouse_LeftDown)) {
    UI_ptr->mouse_LeftDown_X = x;
    UI_ptr->mouse_LeftDown_Y = y;
    UI_ptr->mouse_LeftDown = true;
    //
    UI_ptr->show_reference_object = true;

    // Buffer the GL camera pose
    UI_ptr->GL_camera_Frame_pre.load_frame_mat(UI_ptr->GL_camera_Frame);
  }
  // Right button pressed
  if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN) &&
      (!UI_ptr->mouse_RightDown)) {
    UI_ptr->mouse_RightDown_X = x;
    UI_ptr->mouse_RightDown_Y = y;
    UI_ptr->mouse_RightDown = true;
    //
    UI_ptr->show_reference_object = true;

    // Buffer the GL camera pose
    UI_ptr->GL_camera_Frame_pre.load_frame_mat(UI_ptr->GL_camera_Frame);
  }

  // Left button bounce
  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_UP) &&
      (UI_ptr->mouse_LeftDown)) {
    UI_ptr->mouse_LeftDown = false;
    UI_ptr->show_reference_object = false;
  }
  // Right button bounce
  if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_UP) &&
      (UI_ptr->mouse_RightDown)) {
    UI_ptr->mouse_RightDown = false;
    UI_ptr->show_reference_object = false;
  }
}

void UI_engine::OpenGL_MouseWheelFunction(int button, int dir, int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  // GLUT_WHEEL_DOWN = -1, GLUT_WHEEL_UP = 1;
  if (dir == -1) {
    UI_ptr->GL_camera_Frame.view_distance +=
        UI_parameters::instance()->UI_camera_translation_sensitivity * 5.0f;
    UI_ptr->GL_camera_Frame.transfer_inFrame(
        0, 0,
        -UI_parameters::instance()->UI_camera_translation_sensitivity * 5.0f);
  } else if (dir == 1 && UI_ptr->GL_camera_Frame.view_distance > 0) {
    UI_ptr->GL_camera_Frame.view_distance -=
        UI_parameters::instance()->UI_camera_translation_sensitivity * 5.0f;
    UI_ptr->GL_camera_Frame.transfer_inFrame(
        0, 0,
        +UI_parameters::instance()->UI_camera_translation_sensitivity * 5.0f);
  }
}

void UI_engine::OpenGL_MouseMoveFunction(int x, int y) {
  UI_engine *UI_ptr = UI_engine::instance();

  UI_ptr->mouse_X = x;
  UI_ptr->mouse_Y = y;
}

void screen_shot(string &save_folder, int frame_id) {
  UI_engine *UI_ptr = UI_engine::instance();

  int viewport_width = UI_ptr->main_viewport_width;
  int viewport_height = UI_ptr->main_viewport_height;

  //
  cv::Mat rgba_image_mat;
  cv::Size image_size(viewport_width, viewport_height);
  rgba_image_mat.create(image_size, CV_8UC4);

  GLint bits;
  glGetIntegerv(GL_ALPHA_BITS, &bits);
  // cout << "bits =  " << bits << endl;

  glReadPixels(0, 0, viewport_width, viewport_height, GL_BGRA, GL_UNSIGNED_BYTE,
               rgba_image_mat.data);

  cv::flip(rgba_image_mat, rgba_image_mat, 0);
  for (int offset = 0; offset < viewport_width * viewport_height; offset++) {
    /*
        printf("%d, %d, %d, %d\r\n", (int)rgba_image_mat.data[offset * 4 + 0],
       (int)rgba_image_mat.data[offset * 4 + 1], \
                                (int)rgba_image_mat.data[offset * 4 + 2],
       (int)rgba_image_mat.data[offset * 4 + 3]);
    */
    if (rgba_image_mat.data[offset * 4 + 0] == 0 &&
        rgba_image_mat.data[offset * 4 + 1] == 0 &&
        rgba_image_mat.data[offset * 4 + 2] == 0) {
      rgba_image_mat.data[offset * 4 + 3] = 0x00;
      // rgba_image_mat.data[offset * 4 + 3] = 0xFF;
    }
  }

  char path_str[1024];
  sprintf(path_str, "%s/%08d.png", save_folder.c_str(), frame_id);

  cv::String cv_str(path_str);

  if (cv::imwrite(cv_str, rgba_image_mat) == true) {
#ifdef LOGGING
    LOG_INFO("Save main viewport screenshot: " + std::string(path_str) +
             " successfully.");
#endif

  } else {
#ifdef LOGGING
    LOG_ERROR("Failed to save image " + std::string(path_str));
#endif

    fprintf(stderr,
            "File %s, Line %d, Function %s(), Failed to save image: %s\n",
            __FILE__, __LINE__, __FUNCTION__, path_str);
  }
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif
