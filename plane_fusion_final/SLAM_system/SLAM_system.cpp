
#include "SLAM_system.h"

#include "SLAM_system_settings.h"
#include "Track_engine/Solver_functor.h"
#include "UI_engine/UI_parameters.h"
#include "Log.h"


#if __unix__
#pragma region "SLAM system base class" {
#elif _WIN32
#pragma region(SLAM system base class)
#endif

SLAM_system::SLAM_system() {
  //! Initialize SLAM_system_settings
  SLAM_system_settings::instance()->set_to_default();
}

void SLAM_system::init_parameters(Data_engine *data_engine_ptr) {
  //
  int raw_color_width, raw_color_height;
  int raw_depth_width, raw_depth_height;
  // Get data_engine
  if (data_engine_ptr != nullptr) {
    this->data_engine = data_engine_ptr;

    this->data_engine->get_color_image_size(raw_color_width, raw_color_height);
    SLAM_system_settings::instance()->set_color_image_size(
        raw_color_width, raw_color_height, true);
    this->data_engine->get_depth_image_size(raw_depth_width, raw_depth_height);
    SLAM_system_settings::instance()->set_depth_image_size(
        raw_depth_width, raw_depth_height, true);
  } else {
    fprintf(stderr, "SLAM_system error : invlid data_engine_ptr !\r\n");
    exit(0);
  }

  // Create preprocess engine
  My_Type::Vector2i raw_color_size, raw_depth_size;
  raw_color_size.width = raw_color_width;
  raw_color_size.height = raw_color_height;
  raw_depth_size.width = raw_depth_width;
  raw_depth_size.height = raw_depth_height;
  this->preprocess_engine->init(
      raw_color_size, raw_depth_size,
      SLAM_system_settings::instance()->image_alginment_patch_width,
      SLAM_system_settings::instance()->number_of_hierarchy_layers);

  //
  if (SLAM_system_settings::instance()->enable_plane_module) {
    // this->plane_detector = new Plane_stereoprojection_detector();
    this->plane_detector = new Plane_super_pixel_detector();

    // Plane_super_pixel_detector * temp_ptr =
    // dynamic_cast<Plane_super_pixel_detector *>(this->plane_detector);
    // temp_ptr->init();
    this->plane_detector->init();
  }

  //
  this->feature_detector = new Feature_detector();

  //
  sdkCreateTimer(&this->timer_average);
  sdkResetTimer(&this->timer_average);
}

ProcessingState SLAM_system::process_frames() {
  ProcessingState current_state = this->processing_state;

  if (this->processing_state != STOP_PROCESS) {
    this->pre_estimated_camera_pose.mat = this->estimated_camera_pose.mat;
    this->pre_estimated_camera_pose.synchronize_to_GPU();
  }

  switch (this->processing_state) {
    case ProcessingState::STOP_PROCESS:
      break;
    case ProcessingState::PROCESS_SINGLE_FRAME: {
      this->processing_state = ProcessingState::STOP_PROCESS;
      this->process_one_frame();
      this->mesh_updated = false;
      frame_id++;
      break;
    }
    case ProcessingState::PROCESS_CONTINUOUS_FRAME: {
      this->process_one_frame();
      this->mesh_updated = false;
      frame_id++;
      break;
    }
    default:
      break;
  }

  // End of process.
  static bool first_reach = true;
  if (this->all_data_process_done && first_reach) {
    printf("End of process\n");
#ifdef LOGGING
  LOG_INFO("Finish main thread ------>");
#endif
    first_reach = false;
    this->end_of_process_data();
  }

  return current_state;
}
#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Blank SLAM system" {
#elif _WIN32
#pragma region(Blank SLAM system)
#endif

void Blank_SLAM_system::init() {
#ifdef LOGGING
  LOG_WARNING("Black_SLAM_system, does not process frames.");
#endif
  this->processing_state = ProcessingState::STOP_PROCESS;
  all_data_process_done = true;
}

void Blank_SLAM_system::process_one_frame() {
  printf("Blank_SLAM_system : This module does not process frames\n");
  this->processing_state = ProcessingState::STOP_PROCESS;
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Ground truth SLAM system" {
#elif _WIN32
#pragma region(Ground truth SLAM system)
#endif


//
Ground_truth_SLAM_system::Ground_truth_SLAM_system() {}
Ground_truth_SLAM_system::~Ground_truth_SLAM_system() {
  delete this->preprocess_engine;
  delete this->map_engine;
}

//
void Ground_truth_SLAM_system::init(Data_engine *data_engine_ptr) {
  //
  this->preprocess_engine = new Preprocess_RGBD();
  this->map_engine = new Basic_Voxel_map();

  //
  this->init_parameters(data_engine_ptr);

  //
  this->init_modules();
}
//
void Ground_truth_SLAM_system::init_modules() {
  // Create track engine
  // Void

  //
  this->track_engine->init();
  //
  this->map_engine->init_map();
}

//
void Ground_truth_SLAM_system::process_one_frame() {
  // Load one frame
  bool load_state = this->data_engine->load_next_frame(
      this->timestamp, this->color_mat, this->depth_mat, false);
  if (!load_state) {
    this->processing_state = ProcessingState::STOP_PROCESS;
    all_data_process_done = true;
    return;
  }

  // Raycast point from map
  this->map_engine->raycast_points_from_map(this->estimated_camera_pose.mat,
                                            RaycastMode::RAYCSAT_FOR_TRACKING);

  // Preprocess
  this->preprocess();

  // Track camera pose
  this->track_camera_pose();

  // Update current data to map
  this->update_to_map();
}

//
void Ground_truth_SLAM_system::preprocess() {
  // Preprocess current image/points (generate hierarchical
  // points/normals/weights etc.)
  this->preprocess_engine->preprocess_image(this->color_mat, this->depth_mat);
  // Preprocess model points/normal (generate hierarchical points/normals)
  this->preprocess_engine->preprocess_model_points(
      this->map_engine->dev_model_points, this->map_engine->dev_model_normals);
}

//
void Ground_truth_SLAM_system::track_camera_pose() {
  Trajectory_node ground_truth_trajectory_node;
  bool state = this->data_engine->get_next_ground_truth_camera_pose(
      ground_truth_trajectory_node);

  // Update camera pose
  if (state) {
    this->estimated_camera_pose.load_pose(ground_truth_trajectory_node);
    this->timestamp = ground_truth_trajectory_node.time;
  } else {
    printf("No ground_truth loaded!\n");
    this->processing_state = ProcessingState::STOP_PROCESS;
  }
}

//
void Ground_truth_SLAM_system::update_to_map() {
  //
  this->map_engine->update_map_after_tracking(
      this->estimated_camera_pose,
      this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
      this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0]);
}

// Render scene
void Ground_truth_SLAM_system::generate_render_info(Eigen::Matrix4f view_pose) {
  //
  if (SLAM_system_settings::instance()->generate_mesh_for_visualization) {
    if (!this->mesh_updated) {
      this->generate_mesh();
      this->mesh_updated = true;
    }
  } else {
    this->map_engine->raycast_points_from_map(view_pose,
                                              RaycastMode::RAYCAST_FOR_VIEW);
  }
}

//
void Ground_truth_SLAM_system::end_of_process_data() {
  printf("End of data\n");
  //
  this->generate_mesh();
  this->mesh_of_total_map.compress_mesh();
}

//
void Ground_truth_SLAM_system::generate_mesh() {
  Basic_Voxel_map *Voxel_map_ptr =
      dynamic_cast<Basic_Voxel_map *>(this->map_engine);
  //
  this->mesh_of_total_map.generate_mesh_from_voxel(
      Voxel_map_ptr->voxel_map_ptr->dev_entrise,
      Voxel_map_ptr->voxel_map_ptr->dev_voxel_block_array);
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Basic voxel SLAM system" {
#elif _WIN32
#pragma region(Basic voxel SLAM system)
#endif

#pragma region()
//
Basic_voxel_SLAM_system::Basic_voxel_SLAM_system() {
  // Create modules
  this->preprocess_engine =
      dynamic_cast<Preprocess_RGBD *>(new Preprocess_RGBD());
  this->track_engine = new Basic_ICP_tracker();
  this->map_engine = new Basic_Voxel_map();
}
Basic_voxel_SLAM_system::~Basic_voxel_SLAM_system() {
  delete this->preprocess_engine;
  delete this->track_engine;
  delete this->map_engine;
}

//
void Basic_voxel_SLAM_system::init(Data_engine *data_engine_ptr) {
  //
  this->init_parameters(data_engine_ptr);
  //
  this->init_modules();
}
//
void Basic_voxel_SLAM_system::init_modules() {
  //
  this->track_engine->init();
  //
  this->map_engine->init_map();
}

//
void Basic_voxel_SLAM_system::process_one_frame() {
  // Load one frame (8.0 ms)
  bool load_state = this->data_engine->load_next_frame(
      this->timestamp, this->color_mat, this->depth_mat, false);
  if (!load_state) {
    this->processing_state = ProcessingState::STOP_PROCESS;
    all_data_process_done = true;
    return;
  }

  printf("frame : %d\n", this->frame_id);
  // Raycast point from map (2.71 ms)
  this->map_engine->raycast_points_from_map(this->estimated_camera_pose.mat,
                                            RaycastMode::RAYCSAT_FOR_TRACKING);

  // Preprocess (1.78 ms)
  this->preprocess();

  // Track camera pose  (1.97 ms)
  this->track_camera_pose();

  // Plane module (1.68 ms)
  if (SLAM_system_settings::instance()->enable_plane_module) {
    this->plane_detector->detect_plane(
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0],
        this->estimated_camera_pose.mat);
    /*this->plane_detector->detect_plane(this->preprocess_engine->dev_hierarchy_model_points.data_ptrs[0],
                                                                       this->preprocess_engine->dev_hierarchy_model_normals.data_ptrs[0],
                                                                       this->estimated_camera_pose.mat);*/

    Basic_Voxel_map *map_ptr =
        dynamic_cast<Basic_Voxel_map *>(this->map_engine);
    this->plane_detector->match_planes(
        map_ptr->plane_map_ptr->plane_list.data(),
        map_ptr->plane_map_ptr->plane_counter,
        this->plane_detector->dev_current_plane_labels,
        this->map_engine->dev_model_plane_labels);
  }

  sdkResetTimer(&(this->timer_average));
  sdkStartTimer(&(this->timer_average));

  sdkStopTimer(&(this->timer_average));
  float elapsed_time = sdkGetAverageTimerValue(&(this->timer_average));
  // float elapsed_time = sdkGetTimerValue(&(this->timer_average));
  printf("elapsed_time = %f\n", elapsed_time);

  // Update current data to map (2.0 ms)
  this->update_to_map();

  if (this->frame_id % 20 == 19 && false) {
    Basic_Voxel_map *map_ptr =
        dynamic_cast<Basic_Voxel_map *>(this->map_engine);
    map_ptr->generate_plane_map();

    this->mesh_of_total_map.generate_mesh_from_plane_array(
        map_ptr->plane_map_ptr->plane_list,
        map_ptr->plane_map_ptr->plane_coordinate_list,
        map_ptr->plane_map_ptr->dev_plane_entry_list,
        map_ptr->plane_map_ptr->dev_plane_pixel_array_list,
        map_ptr->voxel_map_ptr->dev_entrise,
        map_ptr->voxel_map_ptr->dev_voxel_block_array);

    //
    map_ptr->plane_map_ptr->generate_planar_block_render_information();
  }

  // Store camera pose (for results evaluation)
  Trajectory_node one_node(this->timestamp, this->estimated_camera_pose.mat);
  this->estimated_trajectory.push_back(one_node);
}

//
void Basic_voxel_SLAM_system::preprocess() {
  //
  this->preprocess_engine->copy_previous_intensity_as_model();
  // Preprocess current image/points (generate hierarchical
  // points/normals/weights etc.)
  this->preprocess_engine->preprocess_image(this->color_mat, this->depth_mat);
  // Preprocess model points/normal (generate hierarchical points/normals)
  this->preprocess_engine->preprocess_model_points(
      this->map_engine->dev_model_points, this->map_engine->dev_model_normals);
}

//
void Basic_voxel_SLAM_system::track_camera_pose() {
  Basic_ICP_tracker *tracker_ptr =
      dynamic_cast<Basic_ICP_tracker *>(this->track_engine);
  tracker_ptr->track_camera_pose(
      this->preprocess_engine->dev_hierarchy_points,
      this->preprocess_engine->dev_hierarchy_model_points,
      this->preprocess_engine->dev_hierarchy_normals,
      this->preprocess_engine->dev_hierarchy_model_normals,
      this->preprocess_engine->dev_hierarchy_intensity,
      this->preprocess_engine->dev_hierarchy_model_intensity,
      this->preprocess_engine->dev_hierarchy_model_gradient,
      this->estimated_camera_pose.mat);

#ifdef COMPILE_DEBUG_CODE
  tracker_ptr->generate_icp_correspondence_lines(
      this->preprocess_engine->dev_hierarchy_points,
      this->preprocess_engine->dev_hierarchy_model_points,
      this->preprocess_engine->dev_hierarchy_normals,
      this->preprocess_engine->dev_hierarchy_model_normals);
#endif
}

//
void Basic_voxel_SLAM_system::update_to_map() {
  if (SLAM_system_settings::instance()->enable_plane_module) {
    this->map_engine->update_map_after_tracking(
        this->estimated_camera_pose,
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0],
        this->plane_detector->dev_current_plane_labels);

    this->map_engine->update_plane_map(this->plane_detector->current_planes,
                                       this->plane_detector->matches);

    //
    if (true) {
    }

  } else {
    this->map_engine->update_map_after_tracking(
        this->estimated_camera_pose,
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0]);
  }
}

// Render scene
void Basic_voxel_SLAM_system::generate_render_info(Eigen::Matrix4f view_pose) {
  //
  if (SLAM_system_settings::instance()->generate_mesh_for_visualization) {
    if (!this->mesh_updated) {
      //
      this->generate_mesh();
      this->mesh_updated = true;
      //
    }
  } else {
    this->map_engine->raycast_points_from_map(view_pose,
                                              RaycastMode::RAYCAST_FOR_VIEW);
  }
}

//
void Basic_voxel_SLAM_system::end_of_process_data() {
  printf("End of data\n");
  //
  this->generate_mesh();
  this->mesh_of_total_map.compress_mesh();

  // Save trajectory
  this->data_engine->save_trajectory(this->estimated_trajectory);
}

//
void Basic_voxel_SLAM_system::generate_mesh() {
  Basic_Voxel_map *map_ptr = dynamic_cast<Basic_Voxel_map *>(this->map_engine);
  //
  this->mesh_of_total_map.generate_mesh_from_voxel(
      map_ptr->voxel_map_ptr->dev_entrise,
      map_ptr->voxel_map_ptr->dev_voxel_block_array);
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

#if __unix__
#pragma region "Submap_SLAM_system" {
#elif _WIN32
#pragma region(Submap_SLAM_system)
#endif

Submap_SLAM_system::Submap_SLAM_system() {
  // Create modules
  this->preprocess_engine =
      dynamic_cast<Preprocess_RGBD *>(new Preprocess_RGBD());
  // this->track_engine = new Basic_ICP_tracker();
  this->track_engine = new Keypoint_ICP_tracker();
  this->map_engine = new Blank_map();
  this->keypoint_associator = new Associator();
  this->plane_associator = new Associator();
}
Submap_SLAM_system::~Submap_SLAM_system() {
  delete this->preprocess_engine;
  delete this->track_engine;
  delete this->map_engine;
  // Release voxel submap
  for (int map_id = 0; map_id < this->submap_ptr_array.size(); map_id++) {
    delete this->submap_ptr_array[map_id];
  }
  // Release feature submap
  for (int map_id = 0; map_id < this->feature_map_ptr_array.size(); map_id++) {
    delete this->feature_map_ptr_array[map_id];
  }
  // Release meshes
  for (int map_id = 0; map_id < this->mesh_ptr_array.size(); map_id++) {
    delete this->mesh_ptr_array[map_id];
  }
  // Release Camera pose
  for (int map_id = 0; map_id < this->submap_pose_array.size(); map_id++) {
    delete this->submap_pose_array[map_id];
  }
}

//
void Submap_SLAM_system::init(Data_engine *data_engine_ptr) {
  //
  this->init_parameters(data_engine_ptr);
  //
  this->init_modules();
}

void Submap_SLAM_system::init_modules() {
  // Initialize render interface (Blank_map)
  this->map_engine->init_map();

  //
  this->track_engine->init();

  // New submap
  Submap_Voxel_map *map_ptr = new Submap_Voxel_map();
  this->submap_ptr_array.push_back(map_ptr);
  this->submap_ptr_array.back()->init_map();
  // New feature map
  Feature_map *feature_map_ptr = new Feature_map();
  this->feature_map_ptr_array.push_back(feature_map_ptr);
  // New submap trajectory
  Trajectory submap_trajectory;
  this->estimated_trajectory_array.push_back(submap_trajectory);
  // New submap pose
  My_pose *pose_ptr = new My_pose();
  this->submap_pose_array.push_back(pose_ptr);

  // Mesh generator
  Mesh_generator *mesh_ptr = new Mesh_generator();
  this->mesh_ptr_array.push_back(mesh_ptr);

  //
  // const string vocabulary_path =
  // "C:/My_SLAM_project/DBoW_result/vocabulary_orb/voc_L3K5_T500N300.yml.gz";
  // this->ORB_vocabulary.load(vocabulary_path);
}

void Submap_SLAM_system::process_one_frame() {
  /* Debug CODE : repeat fuse single frame to fvalidate coordinate alignment */
  /*static bool first_reach = true;
  bool load_state;
  if (first_reach)
  {
  first_reach = false;
  load_state = this->data_engine->load_next_frame(this->timestamp,
  this->color_mat, this->depth_mat, false);
  }*/

  // Load one frame
  bool load_state = this->data_engine->load_next_frame(
      this->timestamp, this->color_mat, this->depth_mat, false);
  if (!load_state) {
    this->processing_state = ProcessingState::STOP_PROCESS;
    all_data_process_done = true;
    return;
  }

  // Raycast point from map
  this->submap_ptr_array.back()->raycast_points_from_map(
      this->estimated_camera_pose.mat, RaycastMode::RAYCSAT_FOR_TRACKING);

  // Preprocess
  this->preprocess();

  // Detect and match features
  this->feature_detector->detect_orb_features(
      this->preprocess_engine->dev_hierarchy_intensity,
      this->preprocess_engine->hierarchy_points);
  this->feature_map_ptr_array.back()->get_model_keypoints(
      this->feature_detector->current_keypoint_position,
      this->estimated_camera_pose,
      this->feature_detector->visible_model_keypoints,
      this->feature_detector->visible_model_features,
      this->feature_detector->visible_point_model_index);
  this->feature_detector->match_orb_features(
      this->feature_map_ptr_array.back()->model_keypoints.size());

  // sdkResetTimer(&(this->timer_average));
  // sdkStartTimer(&(this->timer_average));
  // sdkStopTimer(&(this->timer_average));
  // float elapsed_time = sdkGetAverageTimerValue(&(this->timer_average));
  ////float elapsed_time = sdkGetTimerValue(&(this->timer_average));
  // printf("%f\n", elapsed_time);

  // Track camera pose
  this->track_camera_pose();

  // Plane module (1.68 ms)
  if (SLAM_system_settings::instance()->enable_plane_module && true) {
    this->plane_detector->detect_plane(
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0],
        this->estimated_camera_pose.mat);

    Submap_Voxel_map *map_ptr =
        dynamic_cast<Submap_Voxel_map *>(this->submap_ptr_array.back());
    this->plane_detector->match_planes(
        map_ptr->plane_map_ptr->plane_list.data(),
        map_ptr->plane_map_ptr->plane_counter,
        this->plane_detector->dev_current_plane_labels,
        map_ptr->dev_model_plane_labels);
  }
  // Update current data to map
  this->update_to_map();

  // Store camera pose (for results evaluation)
  Trajectory_node one_node(this->timestamp, this->estimated_camera_pose.mat);
  this->estimated_trajectory_array.back().push_back(one_node);
  // Generate estimated trajectory
  this->estimated_trajectory.clear();
  for (int submap_id = 0; submap_id < this->estimated_trajectory_array.size();
       submap_id++) {
    Eigen::Matrix4f submap_pose = this->submap_pose_array[submap_id]->mat;
    for (int trajectory_node_id = 0;
         trajectory_node_id <
         this->estimated_trajectory_array[submap_id].size();
         trajectory_node_id++) {
      Trajectory_node temp_node =
          this->estimated_trajectory_array[submap_id][trajectory_node_id];
      // TODO : rotation
      // Eigen::Quaternionf rot_q_src(temp_node.quaternions.qr,
      // temp_node.quaternions.qx, temp_node.quaternions.qy,
      // temp_node.quaternions.qz); Eigen::Matrix3f rot_mat =
      // rot_q_src.toRotationMatrix();

      Eigen::Vector3f trans_vec(temp_node.tx, temp_node.ty, temp_node.tz);
      trans_vec = submap_pose.block(0, 0, 3, 3) * trans_vec.eval() +
                  submap_pose.block(0, 3, 3, 1);
      temp_node.tx = trans_vec.x();
      temp_node.ty = trans_vec.y();
      temp_node.tz = trans_vec.z();

      this->estimated_trajectory.push_back(temp_node);
    }
  }
}

//
void Submap_SLAM_system::preprocess() {
  // Preprocess current image/points (generate hierarchical
  // points/normals/weights etc.)
  this->preprocess_engine->preprocess_image(this->color_mat, this->depth_mat);
  // Preprocess model points/normal (generate hierarchical points/normals)
  this->preprocess_engine->preprocess_model_points(
      this->submap_ptr_array.back()->dev_model_points,
      this->submap_ptr_array.back()->dev_model_normals);
}

//
void Submap_SLAM_system::track_camera_pose() {
  // Generate current keypoints and model keypoints
  std::vector<Eigen::Vector3f> current_keypoints;
  std::vector<Eigen::Vector3f> model_keypoints;
  current_keypoints.resize(
      this->feature_detector->current_keypoint_position.size());
  model_keypoints.resize(
      this->feature_detector->current_keypoint_position.size());
  //
  int max_model_number =
      this->feature_map_ptr_array.back()->model_keypoints.size();
  Eigen::Matrix4f pose_inv = this->estimated_camera_pose.mat.inverse();
  for (int current_point_id = 0;
       current_point_id <
       this->feature_detector->current_keypoint_position.size();
       current_point_id++) {
    int model_point_id =
        this->feature_detector->current_match_to_model_id[current_point_id];
    if (model_point_id >= 0 && model_point_id < max_model_number) {
      current_keypoints[current_point_id] = Eigen::Vector3f(
          this->feature_detector->current_keypoint_position[current_point_id].x,
          this->feature_detector->current_keypoint_position[current_point_id].y,
          this->feature_detector->current_keypoint_position[current_point_id]
              .z);
      Eigen::Vector3f model_point(this->feature_map_ptr_array.back()
                                      ->model_keypoints[model_point_id]
                                      .point.x,
                                  this->feature_map_ptr_array.back()
                                      ->model_keypoints[model_point_id]
                                      .point.y,
                                  this->feature_map_ptr_array.back()
                                      ->model_keypoints[model_point_id]
                                      .point.z);
      model_point = pose_inv.block(0, 0, 3, 3) * model_point.eval() +
                    pose_inv.block(0, 3, 3, 1);
      model_keypoints[current_point_id] = model_point;
    } else {
      current_keypoints[current_point_id] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
      model_keypoints[current_point_id] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    }
  }

  // ICP + keypoint tracking
  Keypoint_ICP_tracker *tracker_ptr =
      dynamic_cast<Keypoint_ICP_tracker *>(this->track_engine);
  tracker_ptr->track_camera_pose(
      this->preprocess_engine->dev_hierarchy_points,
      this->preprocess_engine->dev_hierarchy_model_points,
      this->preprocess_engine->dev_hierarchy_normals,
      this->preprocess_engine->dev_hierarchy_model_normals,
      this->preprocess_engine->dev_hierarchy_intensity,
      this->preprocess_engine->dev_hierarchy_model_intensity,
      this->preprocess_engine->dev_hierarchy_model_gradient, current_keypoints,
      model_keypoints, this->estimated_camera_pose.mat);

  this->estimated_camera_pose.synchronize_to_GPU();

#ifdef COMPILE_DEBUG_CODE
  tracker_ptr->generate_icp_correspondence_lines(
      this->preprocess_engine->dev_hierarchy_points,
      this->preprocess_engine->dev_hierarchy_model_points,
      this->preprocess_engine->dev_hierarchy_normals,
      this->preprocess_engine->dev_hierarchy_model_normals);
#endif
}

//
void Submap_SLAM_system::update_to_map() {
  static int static_triangle_offset = 0;

  // Voxel
  this->submap_ptr_array.back()->update_map_after_tracking(
      this->estimated_camera_pose,
      this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
      this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0],
      this->plane_detector->dev_current_plane_labels);
  // Plane
  this->submap_ptr_array.back()->update_plane_map(
      this->plane_detector->current_planes, this->plane_detector->matches);
  // Keypoint
  this->feature_map_ptr_array.back()->update_current_features(
      this->feature_detector->current_keypoint_position,
      this->feature_detector->current_features,
      this->feature_detector->current_match_to_model_id,
      this->feature_detector->previous_match_to_model_id,
      this->estimated_camera_pose);

  // Detect loop closure
  if (frame_id % 10 == 1 && this->enable_loop_detection) {
    detect_loop();
  }
  // Consider to create new submap
  if (this->submap_ptr_array.back()->consider_to_create_new_submap() &&
      this->enable_submap_creation) {
    // this->processing_state = ProcessingState::STOP_PROCESS;

    // Optimization submaps
    if (this->enable_optimization) this->optimize_map();

    // Compress old submap
    this->submap_ptr_array.back()->compress_voxel_map();
    // Generate triagle mesh;
    this->mesh_ptr_array.back()->generate_mesh_from_voxel(
        this->submap_ptr_array.back()->voxel_map_ptr->dev_entrise,
        this->submap_ptr_array.back()->voxel_map_ptr->dev_voxel_block_array);

    // Create new submap
    int last_submap_id = this->submap_ptr_array.size() - 1;
    int current_submap_id = this->submap_ptr_array.size();
    Submap_Voxel_map *last_map_ptr = this->submap_ptr_array.back();
    // -------------------------------- Voxel map
    Submap_Voxel_map *map_ptr = new Submap_Voxel_map();
    this->submap_ptr_array.push_back(map_ptr);
    this->submap_ptr_array.back()->init_map();
    // Update SDF from last submap
    // this->submap_ptr_array.back()->update_map_form_last_map(this->estimated_camera_pose,
    //														this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
    //														last_map_ptr->voxel_map_ptr->dev_entrise,
    //														last_map_ptr->voxel_map_ptr->dev_voxel_block_array);
    this->submap_ptr_array.back()->update_map_after_tracking(
        this->estimated_camera_pose,
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0]);
    // Generate 2D plane map
    if (this->enable_plane_map) {
      last_map_ptr->generate_plane_map();
      this->mesh_ptr_array.back()->generate_nonplanar_mesh_from_voxel(
          last_map_ptr->voxel_map_ptr->dev_entrise,
          last_map_ptr->voxel_map_ptr->dev_voxel_block_array);

      this->mesh_of_total_map.generate_mesh_from_plane_array(
          last_map_ptr->plane_map_ptr->plane_list,
          last_map_ptr->plane_map_ptr->plane_coordinate_list,
          last_map_ptr->plane_map_ptr->dev_plane_entry_list,
          last_map_ptr->plane_map_ptr->dev_plane_pixel_array_list,
          last_map_ptr->voxel_map_ptr->dev_entrise,
          last_map_ptr->voxel_map_ptr->dev_voxel_block_array);
      //
      last_map_ptr->plane_map_ptr->generate_planar_block_render_information();
    }

    // -------------------------------- Plane map
    // Update plane information
    std::vector<std::pair<int, int>> plane_matches;
    // Relabel plane id
    this->plane_detector->match_planes_to_new_map(
        last_map_ptr->plane_map_ptr->plane_list.data(),
        last_map_ptr->plane_map_ptr->plane_counter,
        this->plane_detector->dev_current_plane_labels,
        last_map_ptr->dev_model_plane_labels, plane_matches);
    // this->match_plane_by_parameter(last_map_ptr->plane_map_ptr->plane_list,
    //							   this->submap_ptr_array.back()->plane_map_ptr->plane_list,
    //							   plane_matches);
    // Fusion to map
    this->submap_ptr_array.back()->update_plane_map(
        this->plane_detector->current_planes, this->plane_detector->matches);
    this->submap_ptr_array.back()->update_map_after_tracking(
        this->estimated_camera_pose,
        this->preprocess_engine->dev_hierarchy_points.data_ptrs[0],
        this->preprocess_engine->dev_hierarchy_normals.data_ptrs[0],
        this->plane_detector->dev_current_plane_labels);
    // Raycast
    this->submap_ptr_array.back()->raycast_points_from_map(
        this->estimated_camera_pose.mat, RaycastMode::RAYCSAT_FOR_TRACKING);
    // Build plane association between neighbor submaps
    this->plane_associator->update_matches(
        plane_matches, std::pair<int, int>(last_submap_id, current_submap_id),
        Matches_type::Tracked_Matches);
    // Release old submap planar-voxel space
    // last_map_ptr->release_voxel_map();

    // -------------------------------- Feature map
    // Create new feature map
    Feature_map *feature_map_ptr = new Feature_map();
    this->feature_map_ptr_array.push_back(feature_map_ptr);
    // Update current features to new map
    std::vector<std::pair<int, int>> feature_intermap_matches;
    std::vector<int> init_map_keypoint_id(
        this->feature_detector->current_match_to_model_id.size());
    for (int i = 0, valid_counter = 0; i < init_map_keypoint_id.size(); i++) {
      if (this->feature_detector->current_match_to_model_id[i] >= 0) {
        init_map_keypoint_id[i] = valid_counter;
        valid_counter++;
        feature_intermap_matches.push_back(std::pair<int, int>(
            this->feature_detector->current_match_to_model_id[i],
            init_map_keypoint_id[i]));
      } else
        init_map_keypoint_id[i] = -1;
    }
    this->feature_detector->current_match_to_model_id = init_map_keypoint_id;
    this->feature_map_ptr_array.back()->update_current_features(
        this->feature_detector->current_keypoint_position,
        this->feature_detector->current_features,
        this->feature_detector->current_match_to_model_id,
        this->feature_detector->previous_match_to_model_id,
        this->estimated_camera_pose);
    // Build feature association between neighbor submaps
    this->keypoint_associator->update_matches(
        feature_intermap_matches,
        std::pair<int, int>(last_submap_id, current_submap_id),
        Matches_type::Tracked_Matches);

    // ------------------------------------------ Meshing
    std::vector<std::vector<My_Type::Vector2i>> submap_map_to_global_plane;
    this->generate_submap_to_global_plane_mapper(submap_map_to_global_plane);
    // Copy triangle information to total map mesh
    this->mesh_ptr_array.back()->compress_mesh();
    static_triangle_offset = 0;
    for (int submap_id = 0; submap_id < this->submap_pose_array.size();
         submap_id++) {
      std::vector<My_Type::Vector2i> relabel_list;
      this->generate_submap_to_global_plane_relabel_list(
          submap_map_to_global_plane, submap_id, relabel_list);
      // generate mesh
      this->mesh_of_total_map.copy_triangle_mesh_from(
          this->mesh_ptr_array[submap_id], static_triangle_offset,
          this->submap_pose_array[submap_id]->mat,
          this->submap_ptr_array[submap_id]->voxel_map_ptr->dev_entrise,
          this->submap_ptr_array[submap_id]
              ->voxel_map_ptr->dev_voxel_block_array,
          relabel_list);
      static_triangle_offset +=
          this->mesh_ptr_array[submap_id]->number_of_triangles;
    }
    //
    Mesh_generator *mesh_ptr = new Mesh_generator();
    this->mesh_ptr_array.push_back(mesh_ptr);

    // Init map pose of new submap
    My_pose *pose_ptr = new My_pose();
    this->submap_pose_array.push_back(pose_ptr);

    // New submap trajectory
    Trajectory submap_trajectory;
    this->estimated_trajectory_array.push_back(submap_trajectory);
  }
  // Update current submap mesh
  if (this->need_generate_mesh) {
    std::vector<std::vector<My_Type::Vector2i>> submap_map_to_global_plane;
    this->generate_submap_to_global_plane_mapper(submap_map_to_global_plane);
    std::vector<My_Type::Vector2i> relabel_list;
    this->generate_submap_to_global_plane_relabel_list(
        submap_map_to_global_plane, this->submap_ptr_array.size() - 1,
        relabel_list);

    //
    this->mesh_ptr_array.back()->generate_mesh_from_voxel(
        this->submap_ptr_array.back()->voxel_map_ptr->dev_entrise,
        this->submap_ptr_array.back()->voxel_map_ptr->dev_voxel_block_array,
        relabel_list);
    this->mesh_of_total_map.copy_triangle_mesh_from(this->mesh_ptr_array.back(),
                                                    static_triangle_offset);
  }

  //
  printf("frame_id = %d\n", frame_id);

  if ((this->frame_id % 5 == 4) && this->enable_plane_map) {
    Submap_Voxel_map *map_ptr =
        dynamic_cast<Submap_Voxel_map *>(this->submap_ptr_array.back());
    map_ptr->generate_plane_map();

    this->mesh_of_total_map.generate_mesh_from_plane_array(
        map_ptr->plane_map_ptr->plane_list,
        map_ptr->plane_map_ptr->plane_coordinate_list,
        map_ptr->plane_map_ptr->dev_plane_entry_list,
        map_ptr->plane_map_ptr->dev_plane_pixel_array_list,
        map_ptr->voxel_map_ptr->dev_entrise,
        map_ptr->voxel_map_ptr->dev_voxel_block_array);

    //
    map_ptr->plane_map_ptr->generate_planar_block_render_information();
  }
}

//
void Submap_SLAM_system::optimize_map() {
  //
  this->keypoint_associator->prepare_for_optimization();
  this->plane_associator->prepare_for_optimization();
  //
  // printf("this->keypoint_associator->all_submap_id_pair.size() = %d\n",
  // this->keypoint_associator->all_submap_id_pair.size());
  if (this->keypoint_associator->all_submap_id_pair.size() < 1) return;

  // Build the problem.
  ceres::Problem problem;
  // rx ry rz tx ty tz
  vector<double> pose_array;
  pose_array.resize(6 * this->submap_pose_array.size());
  memset(pose_array.data(), 0x00, pose_array.size() * sizeof(double));

  // for each submap feature matches list
  for (int match_list_index = 0;
       match_list_index < this->keypoint_associator->all_submap_id_pair.size();
       match_list_index++) {
    // index of matched fragment map
    int map_1_index =
        this->keypoint_associator->all_submap_id_pair[match_list_index].first;
    int map_2_index =
        this->keypoint_associator->all_submap_id_pair[match_list_index].second;
    //
    double *pose_1_ptr = (double *)(pose_array.data() + map_1_index * 6);
    double *pose_2_ptr = (double *)(pose_array.data() + map_2_index * 6);

    // for each point pair
    for (int point_pair_id = 0;
         point_pair_id <
         this->keypoint_associator->all_matches[match_list_index].size();
         point_pair_id++) {
      int map_1_point_id = this->keypoint_associator
                               ->all_matches[match_list_index][point_pair_id]
                               .first;
      int map_2_point_id = this->keypoint_associator
                               ->all_matches[match_list_index][point_pair_id]
                               .second;
      //
      My_Type::Vector3f point_1_t, point_2_t;
      point_1_t = this->feature_map_ptr_array[map_1_index]
                      ->model_keypoints[map_1_point_id]
                      .point;
      point_2_t = this->feature_map_ptr_array[map_2_index]
                      ->model_keypoints[map_2_point_id]
                      .point;
      Eigen::Vector3f point_1(point_1_t.x, point_1_t.y, point_1_t.z);
      Eigen::Vector3f point_2(point_2_t.x, point_2_t.y, point_2_t.z);

      // Create cost function
      ceres::CostFunction *cost_function_ptr =
          new ceres::AutoDiffCostFunction<Cost_function_of_feature, 3, 6, 6>(
              new Cost_function_of_feature(point_1, point_2));
      // Add residual block
      problem.AddResidualBlock(cost_function_ptr, new ceres::CauchyLoss(0.05),
                               pose_1_ptr, pose_2_ptr);
    }
  }
  // for each submap plane matches list
  for (int match_list_index = 0;
       match_list_index < this->plane_associator->all_submap_id_pair.size();
       match_list_index++) {
    // index of matched fragment map
    int map_1_index =
        this->plane_associator->all_submap_id_pair[match_list_index].first;
    int map_2_index =
        this->plane_associator->all_submap_id_pair[match_list_index].second;
    //
    double *pose_1_ptr = (double *)(pose_array.data() + map_1_index * 6);
    double *pose_2_ptr = (double *)(pose_array.data() + map_2_index * 6);

    // for each plane pair
    for (int plane_paire_id = 0;
         plane_paire_id <
         this->plane_associator->all_matches[match_list_index].size();
         plane_paire_id++) {
      // Read the matched planes' ID in their fragment map
      int map_1_plane_id =
          this->plane_associator->all_matches[match_list_index][plane_paire_id]
              .first;
      int map_2_plane_id =
          this->plane_associator->all_matches[match_list_index][plane_paire_id]
              .second;

      // printf("(%d, %d) -> (%d, %d)\n", map_1_index, map_1_plane_id,
      // map_2_index, map_2_plane_id);

      // Read planes' parameters (normal vector and distance)
      Eigen::Vector3f normal_1, normal_2;
      normal_1.x() = this->submap_ptr_array[map_1_index]
                         ->plane_map_ptr->plane_list[map_1_plane_id]
                         .nx;
      normal_1.y() = this->submap_ptr_array[map_1_index]
                         ->plane_map_ptr->plane_list[map_1_plane_id]
                         .ny;
      normal_1.z() = this->submap_ptr_array[map_1_index]
                         ->plane_map_ptr->plane_list[map_1_plane_id]
                         .nz;
      float distance_1 = this->submap_ptr_array[map_1_index]
                             ->plane_map_ptr->plane_list[map_1_plane_id]
                             .d;
      normal_2.x() = this->submap_ptr_array[map_2_index]
                         ->plane_map_ptr->plane_list[map_2_plane_id]
                         .nx;
      normal_2.y() = this->submap_ptr_array[map_2_index]
                         ->plane_map_ptr->plane_list[map_2_plane_id]
                         .ny;
      normal_2.z() = this->submap_ptr_array[map_2_index]
                         ->plane_map_ptr->plane_list[map_2_plane_id]
                         .nz;
      float distance_2 = this->submap_ptr_array[map_2_index]
                             ->plane_map_ptr->plane_list[map_2_plane_id]
                             .d;

      // printf("  (%f, %f, %f)  %f\n",
      //	   normal_1.x(), normal_1.y(), normal_1.z(), distance_1);
      // printf("  (%f, %f, %f)  %f\n",
      //	   normal_2.x(), normal_2.y(), normal_2.z(), distance_2);

      // Create cost function
      ceres::CostFunction *cost_function_ptr =
          new ceres::AutoDiffCostFunction<Cost_function_of_adjacent_plane, 4, 6,
                                          6>(
              new Cost_function_of_adjacent_plane(normal_1, normal_2,
                                                  distance_1, distance_2));
      // Add residual block
      problem.AddResidualBlock(cost_function_ptr, NULL, pose_1_ptr, pose_2_ptr);
    }
  }
  // Regular expression
  for (int submap_id = 0; submap_id < this->submap_pose_array.size();
       submap_id++) {
    double *pose_ptr = (double *)(pose_array.data() + submap_id * 6);

    // Create cost function
    float regular_weight = 25;
    regular_weight = 1;

    ceres::CostFunction *cost_function_ptr =
        new ceres::AutoDiffCostFunction<Cost_function_of_regular_constrain, 6,
                                        6>(
            new Cost_function_of_regular_constrain(regular_weight));
    // Add residual block
    problem.AddResidualBlock(cost_function_ptr, NULL, pose_ptr);
  }

  // Set frist fragment map's pose identity
  problem.SetParameterBlockConstant(pose_array.data());

  //
  ceres::Solver::Options option;
  // use SuiteSparse
  option.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  option.linear_solver_type = ceres::SPARSE_SCHUR;
  // output solver information
  option.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  // Run solver
  ceres::Solve(option, &problem, &summary);

  // cout << "Ceres solve done" << endl;

  // Update Ceres result to fragment maps' pose
  for (int submap_id = 0; submap_id < this->submap_pose_array.size();
       submap_id++) {
    double *data_ptr = pose_array.data() + submap_id * 6;

    //
    Eigen::Vector3f rodriguez_vec;
    rodriguez_vec.data()[0] = (float)(data_ptr[0]);
    rodriguez_vec.data()[1] = (float)(data_ptr[1]);
    rodriguez_vec.data()[2] = (float)(data_ptr[2]);
    float theta = rodriguez_vec.norm();

    Eigen::Matrix4f pose_mat;
    // Rotation
    Eigen::Matrix3f rodriguez_cross_mat, rot_mat, identity_mat;
    // Check if norm of rodriguez vector near zero
    if (theta > FLT_EPSILON) {
      rodriguez_vec.normalize();
      // Rotation
      identity_mat.setIdentity();
      rodriguez_cross_mat.setZero();
      rodriguez_cross_mat.data()[1] = +rodriguez_vec.data()[2];  // +gamma
      rodriguez_cross_mat.data()[2] = -rodriguez_vec.data()[1];  // -beta
      rodriguez_cross_mat.data()[3] = -rodriguez_vec.data()[2];  // -gamma
      rodriguez_cross_mat.data()[5] = +rodriguez_vec.data()[0];  // +alpha
      rodriguez_cross_mat.data()[6] = +rodriguez_vec.data()[1];  // +beta
      rodriguez_cross_mat.data()[7] = -rodriguez_vec.data()[0];  // -alpha
      //
      rot_mat = identity_mat + sin(theta) * rodriguez_cross_mat +
                (1 - cos(theta)) * (rodriguez_cross_mat * rodriguez_cross_mat);
    } else {
      // Rotation
      identity_mat.setIdentity();
      rodriguez_cross_mat.setZero();
      rodriguez_cross_mat.data()[1] = +rodriguez_vec.data()[2];  // +gamma
      rodriguez_cross_mat.data()[2] = -rodriguez_vec.data()[1];  // -beta
      rodriguez_cross_mat.data()[3] = -rodriguez_vec.data()[2];  // -gamma
      rodriguez_cross_mat.data()[5] = +rodriguez_vec.data()[0];  // +alpha
      rodriguez_cross_mat.data()[6] = +rodriguez_vec.data()[1];  // +beta
      rodriguez_cross_mat.data()[7] = -rodriguez_vec.data()[0];  // -alpha
      //
      rot_mat = identity_mat + rodriguez_cross_mat;
    }

    // Translation
    Eigen::Vector3f translation_vec;
    translation_vec.data()[0] = (float)(data_ptr[3]);
    translation_vec.data()[1] = (float)(data_ptr[4]);
    translation_vec.data()[2] = (float)(data_ptr[5]);

    //
    pose_mat.setIdentity();
    pose_mat.block(0, 0, 3, 3) = rot_mat;
    pose_mat.block(0, 3, 3, 1) = translation_vec;
    // cout << pose_mat << endl;

    //
    this->submap_pose_array[submap_id]->mat = pose_mat;
    this->submap_pose_array[submap_id]->synchronize_to_GPU();
  }

  // Update camera pose
  std::cout << this->submap_pose_array.back()->mat << std::endl;
  this->estimated_camera_pose.mat = this->submap_pose_array.back()->mat *
                                    this->estimated_camera_pose.mat.eval();
  this->estimated_camera_pose.synchronize_to_GPU();
}

//
void Submap_SLAM_system::end_of_process_data() {
  //
  if (this->enable_optimization) this->optimize_map();
  // Compress last submap
  this->submap_ptr_array.back()->compress_voxel_map();

  //
  std::vector<std::vector<My_Type::Vector2i>> submap_map_to_global_plane;
  this->generate_submap_to_global_plane_mapper(submap_map_to_global_plane);
  int static_triangle_offset = 0;
  for (int submap_id = 0; submap_id < this->submap_pose_array.size();
       submap_id++) {
    std::vector<My_Type::Vector2i> relabel_list;
    this->generate_submap_to_global_plane_relabel_list(
        submap_map_to_global_plane, submap_id, relabel_list);
    // generate mesh
    this->mesh_of_total_map.copy_triangle_mesh_from(
        this->mesh_ptr_array[submap_id], static_triangle_offset,
        this->submap_pose_array[submap_id]->mat,
        this->submap_ptr_array[submap_id]->voxel_map_ptr->dev_entrise,
        this->submap_ptr_array[submap_id]->voxel_map_ptr->dev_voxel_block_array,
        relabel_list);
    static_triangle_offset +=
        this->mesh_ptr_array[submap_id]->number_of_triangles;

    // Generate 2D plane map
    if (true) {
      this->submap_ptr_array[submap_id]->generate_plane_map();
      this->mesh_ptr_array.back()->generate_nonplanar_mesh_from_voxel(
          this->submap_ptr_array[submap_id]->voxel_map_ptr->dev_entrise,
          this->submap_ptr_array[submap_id]
              ->voxel_map_ptr->dev_voxel_block_array);

      this->mesh_of_total_map.generate_mesh_from_plane_array(
          this->submap_ptr_array[submap_id]->plane_map_ptr->plane_list,
          this->submap_ptr_array[submap_id]
              ->plane_map_ptr->plane_coordinate_list,
          this->submap_ptr_array[submap_id]
              ->plane_map_ptr->dev_plane_entry_list,
          this->submap_ptr_array[submap_id]
              ->plane_map_ptr->dev_plane_pixel_array_list,
          this->submap_ptr_array[submap_id]->voxel_map_ptr->dev_entrise,
          this->submap_ptr_array[submap_id]
              ->voxel_map_ptr->dev_voxel_block_array);
      //
      this->submap_ptr_array[submap_id]
          ->plane_map_ptr->generate_planar_block_render_information();
    }
  }

  // Regenerate trajectory
  this->estimated_trajectory.clear();
  for (int submap_id = 0; submap_id < this->estimated_trajectory_array.size();
       submap_id++) {
    Eigen::Matrix4f submap_pose = this->submap_pose_array[submap_id]->mat;
    for (int trajectory_node_id = 0;
         trajectory_node_id <
         this->estimated_trajectory_array[submap_id].size();
         trajectory_node_id++) {
      Trajectory_node temp_node =
          this->estimated_trajectory_array[submap_id][trajectory_node_id];
      // Eigen::Quaternionf rot_q_src(temp_node.quaternions.qr,
      // temp_node.quaternions.qx, temp_node.quaternions.qy,
      // temp_node.quaternions.qz); Eigen::Matrix3f rot_mat =
      // rot_q_src.toRotationMatrix();

      Eigen::Vector3f trans_vec(temp_node.tx, temp_node.ty, temp_node.tz);
      trans_vec = submap_pose.block(0, 0, 3, 3) * trans_vec.eval() +
                  submap_pose.block(0, 3, 3, 1);
      temp_node.tx = trans_vec.x();
      temp_node.ty = trans_vec.y();
      temp_node.tz = trans_vec.z();

      this->estimated_trajectory.push_back(temp_node);
    }
  }
  // Save trajectory
  this->data_engine->save_trajectory(this->estimated_trajectory);
}

//
void Submap_SLAM_system::generate_render_info(Eigen::Matrix4f view_pose) {
  //
  this->submap_ptr_array.back()->raycast_points_from_map(
      view_pose, RaycastMode::RAYCAST_FOR_VIEW);

  //
  My_Type::Vector2i main_viewport_size =
      UI_parameters::instance()->main_viewport_size;
  //
  memcpy(this->map_engine->scene_points,
         this->submap_ptr_array.back()->scene_points,
         main_viewport_size.width * main_viewport_size.height *
             sizeof(My_Type::Vector3f));
  memcpy(this->map_engine->scene_normals,
         this->submap_ptr_array.back()->scene_normals,
         main_viewport_size.width * main_viewport_size.height *
             sizeof(My_Type::Vector3f));
}

//
void Submap_SLAM_system::generate_mesh() {
  Submap_Voxel_map *map_ptr = submap_ptr_array.back();

  this->mesh_ptr_array.back()->generate_mesh_from_voxel(
      map_ptr->voxel_map_ptr->dev_entrise,
      map_ptr->voxel_map_ptr->dev_voxel_block_array);
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
//
void Submap_SLAM_system::detect_loop() {
  // Save keyframe
  this->feature_map_ptr_array.back()->save_keyframe(
      this->feature_detector->current_features,
      this->feature_detector->current_match_to_model_id, this->ORB_vocabulary);

  // this->keypoint_buffer_1.clear();
  // std::vector<int> & test_mapper =
  // this->feature_map_ptr_array.back()->keyframe_feature_mapper_list.back();
  // for (int i = 0; i < test_mapper.size(); i++)
  //{
  //	int kp_id = test_mapper[i];
  //	if (kp_id < 0)	continue;
  //	keypoint_buffer_1.push_back(Eigen::Vector3f(this->feature_map_ptr_array.back()->model_keypoints[kp_id].point.x,
  //		this->feature_map_ptr_array.back()->model_keypoints[kp_id].point.y,
  // this->feature_map_ptr_array.back()->model_keypoints[kp_id].point.z));
  //}

  this->keypoint_associator->prepare_for_optimization();
  this->plane_associator->prepare_for_optimization();
  //
  if (this->feature_map_ptr_array.size() < 2) return;
  if (this->keypoint_associator->all_submap_id_pair.size() < 1) return;

  //
  int this_submap_id = this->feature_map_ptr_array.size() - 1;
  int this_keyframe_id = this->feature_map_ptr_array[this_submap_id]
                             ->keyframe_feature_mapper_list.size() -
                         1;
  My_Type::Vector3f this_weight_center =
      this->feature_map_ptr_array[this_submap_id]
          ->keyframe_weigth_centers[this_keyframe_id];
  DBoW3::BowVector current_bowvec = this->feature_map_ptr_array[this_submap_id]
                                        ->dbow_vec_list[this_keyframe_id];

  // Check with in-associated submaps
  this->keypoint_associator->prepare_for_optimization();
  // Build associated flag
  std::vector<bool> associated_flag(this->feature_map_ptr_array.size(), false);
  associated_flag[this_submap_id] = true;
  for (int pair_id = 0;
       pair_id < this->keypoint_associator->all_submap_id_pair.size();
       pair_id++) {
    if (this->keypoint_associator->all_submap_id_pair[pair_id].first ==
        this_submap_id) {
      associated_flag[this->keypoint_associator->all_submap_id_pair[pair_id]
                          .second] = true;
    }
    if (this->keypoint_associator->all_submap_id_pair[pair_id].second ==
        this_submap_id) {
      associated_flag[this->keypoint_associator->all_submap_id_pair[pair_id]
                          .first] = true;
    }
  }
  //
  int matched_keyframe_id = -1, matched_submap_id = -1, similarity_counter = 0;
  float max_similarity = 0;
  for (int submap_id = 0; submap_id < this->feature_map_ptr_array.size();
       submap_id++) {
    if (associated_flag[submap_id]) continue;

    for (int keyframe_id = 0;
         keyframe_id < this->feature_map_ptr_array[submap_id]
                           ->keyframe_feature_mapper_list.size();
         keyframe_id++) {
      My_Type::Vector3f check_weight_center =
          this->feature_map_ptr_array[submap_id]
              ->keyframe_weigth_centers[keyframe_id];
      // if ((check_weight_center - this_weight_center).norm() > 1.0f)
      // continue;
      if ((check_weight_center - this_weight_center).norm() > 2.0f) continue;
      if (this->feature_map_ptr_array[submap_id]
              ->keyframe_feature_mapper_list[keyframe_id]
              .size() < 10)
        continue;

      // Check for DBoW3 ORB feature similarity
      float similar_ratio = 0.0f;
      DBoW3::BowVector keyframe_bowvec =
          this->feature_map_ptr_array[submap_id]->dbow_vec_list[keyframe_id];

      similar_ratio =
          (float)this->ORB_vocabulary.score(current_bowvec, keyframe_bowvec);
      cout << "similar_ratio = " << similar_ratio << endl;
      // Save canditate looped keyframe for RANSAC re-check

      max_similarity = fmaxf(max_similarity, similar_ratio);
      if (similar_ratio > 0.40f) {
        max_similarity = similar_ratio;
        matched_submap_id = submap_id;
        matched_keyframe_id = keyframe_id;
        similarity_counter++;

        // printf("%d, %d -> %d, %d\n", this_submap_id, this_keyframe_id,
        // matched_submap_id, matched_keyframe_id); std::cout << "similar_ratio
        // = " << similar_ratio << std::endl;
      }
    }
  }
  // printf("max_similarity %f\n", max_similarity);

  // Solve loop
  if (similarity_counter > 5) {
    // Generate loop feature / loop keypoints
    // this->feature_map_ptr_array[this_submap_id]->model_keypoints[]
    cv::Mat loop_features;
    // this->feature_detector->current_keypoint_position
    std::vector<My_Type::Vector3f> loop_keypoint_position;
    //
    std::vector<std::pair<int, int>> current_keypoint_mapper;
    std::vector<std::pair<int, int>> loop_keypoint_mapper;
    // Current info
    for (int current_id = 0;
         current_id < this->feature_detector->current_match_to_model_id.size();
         current_id++) {
      current_keypoint_mapper.push_back(std::pair<int, int>(
          this->submap_ptr_array.size() - 1,
          this->feature_detector->current_match_to_model_id[current_id]));
    }
    // Looped info
    std::vector<int> &matched_feature_mapper =
        this->feature_map_ptr_array[matched_submap_id]
            ->keyframe_feature_mapper_list[matched_keyframe_id];
    for (int mapper_id = 0; mapper_id < matched_feature_mapper.size();
         mapper_id++) {
      int model_keypoint_id = matched_feature_mapper[mapper_id];
      if (model_keypoint_id < 0) continue;

      //
      int valid_scale_id = -1;
      for (int scale_id = 0; scale_id < MAX_LAYER_NUMBER; scale_id++) {
        if (this->feature_map_ptr_array[matched_submap_id]
                ->model_feature_scale_flag[model_keypoint_id]
                .exist[scale_id] != 0) {
          valid_scale_id = scale_id;
          break;
        }
      }

      if (valid_scale_id >= 0 && model_keypoint_id >= 0) {
        loop_features.push_back(this->feature_map_ptr_array[matched_submap_id]
                                    ->model_features[model_keypoint_id]
                                    .row(valid_scale_id));
        loop_keypoint_position.push_back(
            this->feature_map_ptr_array[matched_submap_id]
                ->model_keypoints[model_keypoint_id]
                .point);
        loop_keypoint_mapper.push_back(
            std::pair<int, int>(matched_submap_id, model_keypoint_id));
      }
    }

    // Associate looped keypoints and planes
    // -keypoint
    std::vector<Eigen::Vector3f> current_position_buffer, loop_position_buffer;
    std::vector<std::pair<int, int>>
        keypoint_matches; /* (current keypoint -> looped keypoint) */
    std::vector<bool> is_valid_keypoint_match;
    for (int current_id = 0;
         current_id < this->feature_detector->current_match_to_model_id.size();
         current_id++) {
      int current_keypoint_index =
          this->feature_detector->current_match_to_model_id[current_id];
      if (current_keypoint_index < 0) continue;

      My_Type::Vector3f current_position =
          this->feature_map_ptr_array[this_submap_id]
              ->model_keypoints[current_keypoint_index]
              .point;
      cv::Mat current_feature_row =
          this->feature_detector->current_features.row(current_id);
      // cv::Mat current_feature_row =
      // this->feature_map_ptr_array[this_submap_id]->model_features[current_keypoint_index].row;

      for (int loop_id = 0; loop_id < loop_keypoint_position.size();
           loop_id++) {
        My_Type::Vector3f loop_position = loop_keypoint_position[loop_id];
        cv::Mat loop_feature_row = loop_features.row(loop_id);

        if ((current_position - loop_position).norm() > 0.25) continue;
        if (DescriptorDistance(current_feature_row, loop_feature_row) > 64)
          continue;

        //
        current_position_buffer.push_back(Eigen::Vector3f(
            current_position.x, current_position.y, current_position.z));
        loop_position_buffer.push_back(
            Eigen::Vector3f(loop_position.x, loop_position.y, loop_position.z));
        keypoint_matches.push_back(
            std::pair<int, int>(current_keypoint_index, loop_id));
        // printf("%f, %d -> %d\n", (current_position - loop_position).norm(),
        // current_keypoint_index, loop_id);
        is_valid_keypoint_match.push_back(true);
      }
    }
    // -plane
    std::vector<Plane_info> current_plane_buffer, loop_plane_buffer;
    std::vector<std::pair<int, int>>
        plane_matches; /* (current plane -> looped plane) */
    std::vector<bool> is_valid_plane_match;
    for (int current_id = 0;
         current_id <
         this->submap_ptr_array.back()->plane_map_ptr->plane_list.size();
         current_id++) {
      if (!this->submap_ptr_array.back()
               ->plane_map_ptr->plane_list[current_id]
               .is_valid)
        continue;
      My_Type::Vector3f current_normal;
      current_normal.nx = this->submap_ptr_array.back()
                              ->plane_map_ptr->plane_list[current_id]
                              .nx;
      current_normal.ny = this->submap_ptr_array.back()
                              ->plane_map_ptr->plane_list[current_id]
                              .ny;
      current_normal.nz = this->submap_ptr_array.back()
                              ->plane_map_ptr->plane_list[current_id]
                              .nz;
      float current_distance = this->submap_ptr_array.back()
                                   ->plane_map_ptr->plane_list[current_id]
                                   .d;

      for (int loop_id = 0; loop_id < this->submap_ptr_array[matched_submap_id]
                                          ->plane_map_ptr->plane_list.size();
           loop_id++) {
        if (!this->submap_ptr_array[matched_submap_id]
                 ->plane_map_ptr->plane_list[loop_id]
                 .is_valid)
          continue;
        My_Type::Vector3f loop_normal;
        loop_normal.nx = this->submap_ptr_array[matched_submap_id]
                             ->plane_map_ptr->plane_list[loop_id]
                             .nx;
        loop_normal.ny = this->submap_ptr_array[matched_submap_id]
                             ->plane_map_ptr->plane_list[loop_id]
                             .ny;
        loop_normal.nz = this->submap_ptr_array[matched_submap_id]
                             ->plane_map_ptr->plane_list[loop_id]
                             .nz;
        float loop_distance = this->submap_ptr_array[matched_submap_id]
                                  ->plane_map_ptr->plane_list[loop_id]
                                  .d;

        if (current_normal.dot(loop_normal) < 0.90f) continue;
        if (fabsf(current_distance - loop_distance) > 0.15f) continue;

        current_plane_buffer.push_back(
            this->submap_ptr_array.back()
                ->plane_map_ptr->plane_list[current_id]);
        loop_plane_buffer.push_back(this->submap_ptr_array[matched_submap_id]
                                        ->plane_map_ptr->plane_list[loop_id]);
        plane_matches.push_back(std::pair<int, int>(current_id, loop_id));
        is_valid_plane_match.push_back(true);
      }
    }

    //// RANSAC solve associations
    // Eigen::Matrix4f test_pose, best_pose;
    // int test_valid_pair, best_valid_pair = 0;

    // Iterate optimization
    // this->filter_loop_matches(current_position_buffer, loop_position_buffer,
    //						  current_plane_buffer,
    // loop_plane_buffer,
    // is_valid_keypoint_match, is_valid_plane_match);

    keypoint_buffer_1.clear();
    for (int i = 0;
         i < this->feature_detector->current_match_to_model_id.size(); i++) {
      int point_id = this->feature_detector->current_match_to_model_id[i];
      if (point_id < 0) continue;

      keypoint_buffer_1.push_back(
          Eigen::Vector3f(this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[point_id]
                              .point.x,
                          this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[point_id]
                              .point.y,
                          this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[point_id]
                              .point.z));
    }

    keypoint_buffer_2.clear();
    for (int i = 0; i < this->feature_map_ptr_array[matched_submap_id]
                            ->keyframe_feature_mapper_list[matched_keyframe_id]
                            .size();
         i++) {
      std::vector<int> &temp_mapper =
          this->feature_map_ptr_array[matched_submap_id]
              ->keyframe_feature_mapper_list[matched_keyframe_id];
      if (temp_mapper[i] < 0) continue;

      keypoint_buffer_2.push_back(
          Eigen::Vector3f(this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[temp_mapper[i]]
                              .point.x,
                          this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[temp_mapper[i]]
                              .point.y,
                          this->feature_map_ptr_array[matched_submap_id]
                              ->model_keypoints[temp_mapper[i]]
                              .point.z));
    }

    // Associate planes
    for (int plane_pair_id = 0; plane_pair_id < plane_matches.size();
         plane_pair_id++) {
      Plane_info current_plane =
          this->submap_ptr_array[this_submap_id]
              ->plane_map_ptr->plane_list[plane_matches[plane_pair_id].first];
      Plane_info loop_plane =
          this->submap_ptr_array[matched_submap_id]
              ->plane_map_ptr->plane_list[plane_matches[plane_pair_id].second];

      My_Type::Vector3f current_normal(current_plane.nx, current_plane.ny,
                                       current_plane.nz);
      float current_distance = current_plane.d;
      My_Type::Vector3f loop_normal(loop_plane.nx, loop_plane.ny,
                                    loop_plane.nz);
      float loop_distance = loop_plane.d;

      // printf("%f, %f\n", current_normal.dot(loop_normal),
      // fabsf(current_distance - loop_distance));
    }

    //
    this->keypoint_associator->update_matches(
        std::vector<std::pair<int, int>>(),
        std::pair<int, int>(this_submap_id, matched_submap_id),
        Matches_type::Relocalization_Matches);
    this->plane_associator->update_matches(
        plane_matches, std::pair<int, int>(this_submap_id, matched_submap_id),
        Matches_type::Relocalization_Matches);
  }
}

//
int find_submap_plane_in_list(
    std::vector<std::vector<My_Type::Vector2i>> &global_plane_container,
    const My_Type::Vector2i &submap_plane) {
  for (int plane_id = 0; plane_id < global_plane_container.size(); plane_id++) {
    for (int list_id = 0; list_id < global_plane_container[plane_id].size();
         list_id++) {
      if (global_plane_container[plane_id][list_id] == submap_plane)
        return plane_id;
    }
  }
  return -1;
}
//
void Submap_SLAM_system::generate_submap_to_global_plane_mapper(
    std::vector<std::vector<My_Type::Vector2i>> &global_plane_container) {
  global_plane_container.clear();
  global_plane_container.push_back(std::vector<My_Type::Vector2i>());
  //
  this->plane_associator->prepare_for_optimization();

  // Associated planes
  for (int map_match_id = 0;
       map_match_id < this->plane_associator->all_submap_id_pair.size();
       map_match_id++) {
    for (int pair_id = 0;
         pair_id < this->plane_associator->all_matches[map_match_id].size();
         pair_id++) {
      My_Type::Vector2i first_plane(
          this->plane_associator->all_submap_id_pair[map_match_id].first,
          this->plane_associator->all_matches[map_match_id][pair_id].first);
      My_Type::Vector2i second_plane(
          this->plane_associator->all_submap_id_pair[map_match_id].second,
          this->plane_associator->all_matches[map_match_id][pair_id].second);

      int first_global_id =
          find_submap_plane_in_list(global_plane_container, first_plane);
      int second_global_id =
          find_submap_plane_in_list(global_plane_container, second_plane);
      if (first_global_id == -1 && second_global_id == -1) {
        // Add new global plane
        global_plane_container.push_back(std::vector<My_Type::Vector2i>());
        global_plane_container.back().push_back(first_plane);
        global_plane_container.back().push_back(second_plane);
      } else if (first_global_id != -1 && second_global_id != -1 &&
                 first_global_id != second_global_id) {
        // Merge two global plane
        std::vector<std::vector<My_Type::Vector2i>> buffer_container;
        for (int global_plane_id = 0;
             global_plane_id < global_plane_container.size();
             global_plane_id++) {
          if (first_global_id != global_plane_id &&
              second_global_id != global_plane_id)
            buffer_container.push_back(global_plane_container[global_plane_id]);
        }
        buffer_container.push_back(global_plane_container[first_global_id]);
        buffer_container.back().insert(
            buffer_container.back().begin(),
            global_plane_container[second_global_id].begin(),
            global_plane_container[second_global_id].end());
        global_plane_container = buffer_container;
      } else if (first_global_id != -1 && second_global_id == -1) {
        global_plane_container[first_global_id].push_back(second_plane);
      } else if (first_global_id == -1 && second_global_id != -1) {
        global_plane_container[second_global_id].push_back(first_plane);
      }

      //{
      //	// Link to existed global plane
      //	int global_plane_id = std::max(first_global_id,
      // second_global_id); 	if (first_global_id != -1)
      // global_plane_container[global_plane_id].push_back(second_plane);
      // if (second_global_id != -1)
      // global_plane_container[global_plane_id].push_back(first_plane);
      //}
    }
  }

  // Independent planes
  for (int submap_id = 0; submap_id < this->submap_ptr_array.size();
       submap_id++) {
    for (int plane_id = 1;
         plane_id <
         this->submap_ptr_array[submap_id]->plane_map_ptr->plane_list.size();
         plane_id++) {
      if (!this->submap_ptr_array[submap_id]
               ->plane_map_ptr->plane_list[plane_id]
               .is_valid)
        continue;
      if (find_submap_plane_in_list(global_plane_container,
                                    My_Type::Vector2i(submap_id, plane_id)) ==
          -1) {
        std::vector<My_Type::Vector2i> temp_vec;
        temp_vec.push_back(My_Type::Vector2i(submap_id, plane_id));
        global_plane_container.push_back(temp_vec);
      }
    }
  }

  if (false) {
    for (int global_id = 0; global_id < global_plane_container.size();
         global_id++) {
      printf("%d: ", global_id);
      for (int pair_id = 0; pair_id < global_plane_container[global_id].size();
           pair_id++) {
        printf("(%d, %d)", global_plane_container[global_id][pair_id].x,
               global_plane_container[global_id][pair_id].y);
      }
      printf("\n");
    }
    printf("\n");
  }
}

//
void Submap_SLAM_system::generate_submap_to_global_plane_relabel_list(
    std::vector<std::vector<My_Type::Vector2i>> &global_plane_container,
    int submap_id, std::vector<My_Type::Vector2i> &relabel_list) {
  //
  std::vector<My_Type::Vector2i> temp_list;
  for (int global_id = 0; global_id < global_plane_container.size();
       global_id++)
    for (int pair_id = 0; pair_id < global_plane_container[global_id].size();
         pair_id++)
      if (global_plane_container[global_id][pair_id].x == submap_id)
        temp_list.push_back(My_Type::Vector2i(
            global_id, global_plane_container[global_id][pair_id].y));

  //
  int max_plane_label = 0;
  for (int i = 0; i < temp_list.size(); i++)
    max_plane_label = std::max(max_plane_label, temp_list[i].y);

  //
  relabel_list.clear();
  relabel_list.resize(max_plane_label + 1);
  memset(relabel_list.data(), 0x00,
         relabel_list.size() * sizeof(My_Type::Vector2i));
  for (int list_id = 0; list_id < relabel_list.size(); list_id++)
    relabel_list[list_id].x = list_id;
  for (int global_id = 0; global_id < temp_list.size(); global_id++)
    relabel_list[temp_list[global_id].y].y = temp_list[global_id].x;
}

void Submap_SLAM_system::filter_loop_matches(
    std::vector<Eigen::Vector3f> &current_points,
    std::vector<Eigen::Vector3f> &loop_points,
    std::vector<Plane_info> &current_planes,
    std::vector<Plane_info> &loop_planes,
    std::vector<bool> &is_valid_keypoint_match,
    std::vector<bool> &is_valid_plane_match) {
  //!
  const float huber_radius = 0.05f;
  const float valid_radius = 0.35f;
  const float huber_plane_normal = 0.9f;
  const float huber_plane_distance = 0.1f;
  const float valid_plane_normal = 0.8f;
  const float valid_plane_distance = 0.2f;
  //
  Eigen::Matrix4f pose;
  pose.setIdentity();

  //
  LevenbergMarquardt_solver my_solver;

  // Get better pose
  for (int iterate_i = 0; iterate_i < 20; iterate_i++) {
    //
    Eigen::MatrixXf hessian_total(6, 6), nabla_total(6, 1);
    Eigen::MatrixXf hessian_keypoint(6, 6), nabla_keypoint(6, 1);
    Eigen::MatrixXf hessian_plane(6, 6), nabla_plane(6, 1);

    // Keypoints
    hessian_keypoint.setZero();
    nabla_keypoint.setZero();
    int valid_keypoint_number = 0;
    float point_diff_avg = 0;
    for (int point_id = 0; point_id < current_points.size(); point_id++) {
      Eigen::Vector3f current_point_W =
          pose.block(0, 0, 3, 3) * current_points[point_id] +
          pose.block(0, 3, 3, 1);
      Eigen::Vector3f loop_point = loop_points[point_id];

      float point_diff = (current_point_W - loop_point).norm();
      if (point_diff > valid_radius) continue;

      Eigen::MatrixXf jacobian_mat(3, 6), b_mat(3, 1);
      jacobian_mat.setZero();
      jacobian_mat.block(0, 3, 3, 3).setIdentity();
      jacobian_mat.data()[1] = -current_point_W.z();
      jacobian_mat.data()[2] = +current_point_W.y();
      jacobian_mat.data()[6] = -current_point_W.y();
      jacobian_mat.data()[3] = +current_point_W.z();
      jacobian_mat.data()[5] = -current_point_W.x();
      jacobian_mat.data()[7] = +current_point_W.x();
      b_mat = current_point_W - loop_point;

      // Huber loss
      if (point_diff < huber_radius) {
        hessian_keypoint += jacobian_mat.transpose() * jacobian_mat;
        nabla_keypoint += jacobian_mat.transpose() * b_mat;
        point_diff_avg += point_diff;
        valid_keypoint_number++;
      } else if (point_diff < valid_radius) {
        // hessian_keypoint += 0;
        Eigen::MatrixXf clamp_b_mat(3, 1);
        clamp_b_mat = b_mat;
        for (int i = 0; i < 3; i++)
          clamp_b_mat.data()[i] =
              fmaxf(fminf(clamp_b_mat.data()[i], huber_radius), -huber_radius);
        nabla_keypoint += jacobian_mat.transpose() * clamp_b_mat;
        point_diff_avg += huber_radius;
        valid_keypoint_number++;
      } else {
        /* Invalid residual */
      }
    }
    point_diff_avg /= valid_keypoint_number;

    // Planes
    int valid_plane_number = 0;
    float plane_diff_avg = 0;
    for (int plane_id = 0; plane_id < current_planes.size(); plane_id++) {
      Eigen::Vector3f current_normal_C;
      current_normal_C.x() = current_planes[plane_id].nx;
      current_normal_C.y() = current_planes[plane_id].ny;
      current_normal_C.z() = current_planes[plane_id].nz;
      float current_d_C = current_planes[plane_id].d;
      //
      Eigen::Vector3f current_normal_W;
      current_normal_W = pose.block(0, 0, 3, 3) * current_normal_C;
      //
      Eigen::Vector3f current_plane_W;
      current_plane_W =
          -current_normal_W * current_d_C + pose.block(0, 3, 3, 1);
      float current_d_W = -current_plane_W.dot(current_normal_W);

      // float normal_
    }

    // Update pose
    int valid_number = valid_keypoint_number + valid_plane_number * 16;
    int loss_total = point_diff_avg + plane_diff_avg * 16;
    hessian_total = hessian_keypoint + hessian_plane * 16;
    nabla_total = nabla_keypoint + nabla_plane * 16;
    my_solver(hessian_total, nabla_total, loss_total, valid_number, pose);

    // printf("loss_total = %f\n", loss_total);
  }

  // Eliminate outlyers
  for (int point_id = 0; point_id < current_points.size(); point_id++) {
    Eigen::Vector3f current_point_W =
        pose.block(0, 0, 3, 3) * current_points[point_id] +
        pose.block(0, 3, 3, 1);
    Eigen::Vector3f loop_point = loop_points[point_id];

    float point_diff = (current_point_W - loop_point).norm();
    if (point_diff > valid_radius) {
      is_valid_keypoint_match[point_id] = false;
    }
  }
  for (int plane_id = 0; plane_id < current_planes.size(); plane_id++) {
    Eigen::Vector3f current_normal_C;
    current_normal_C.x() = current_planes[plane_id].nx;
    current_normal_C.y() = current_planes[plane_id].ny;
    current_normal_C.z() = current_planes[plane_id].nz;
    float current_d_C = current_planes[plane_id].d;
    //
    Eigen::Vector3f current_normal_W;
    current_normal_W = pose.block(0, 0, 3, 3) * current_normal_C;
    //
    Eigen::Vector3f current_plane_W;
    current_plane_W = -current_normal_W * current_d_C + pose.block(0, 3, 3, 1);
    float current_d_W = -current_plane_W.dot(current_normal_W);

    //
    Eigen::Vector3f loop_normal;
    loop_normal.x() = loop_planes[plane_id].nx;
    loop_normal.y() = loop_planes[plane_id].ny;
    loop_normal.z() = loop_planes[plane_id].nz;
    float loop_d = loop_planes[plane_id].d;

    if (current_normal_W.dot(loop_normal) < valid_plane_normal ||
        fabsf(current_d_W - loop_d) > valid_plane_distance) {
      is_valid_plane_match[plane_id] = false;
    }
  }
}

//
bool compute_plane_similarity(Plane_info &plane_1, Plane_info &plane_2) {
  const float inner_product_threshold = 0.95;
  const float distance = 0.04;

  My_Type::Vector3f normal_1(plane_1.nx, plane_1.ny, plane_1.nz);
  My_Type::Vector3f normal_2(plane_2.nx, plane_2.ny, plane_2.nz);
  if (normal_1.dot(normal_2) < inner_product_threshold) return false;
  if (fabsf(plane_1.d - plane_2.d) > distance) return false;

  return true;
}
//
void Submap_SLAM_system::match_plane_by_parameter(
    std::vector<Plane_info> &previous_map_planes,
    std::vector<Plane_info> &current_map_planes,
    std::vector<std::pair<int, int>> &plane_matches) {
  //
  for (int pre_plane_id = 0; pre_plane_id < previous_map_planes.size();
       pre_plane_id++) {
    if (!previous_map_planes[pre_plane_id].is_valid) continue;
    bool already_matched = false;
    for (int matched_id = 0; matched_id < plane_matches.size(); matched_id++) {
      if (plane_matches[matched_id].first == pre_plane_id) {
        already_matched = true;
        break;
      }
    }
    if (already_matched) continue;

    for (int current_plane_id = 0; current_plane_id < current_map_planes.size();
         current_plane_id++) {
      if (!current_map_planes[current_plane_id].is_valid) continue;

      bool is_similar_plane =
          compute_plane_similarity(previous_map_planes[pre_plane_id],
                                   current_map_planes[current_plane_id]);

      if (is_similar_plane) {
        plane_matches.push_back(
            std::pair<int, int>(pre_plane_id, current_plane_id));
        // printf("%d => %d\n", pre_plane_id, current_plane_id);
        break;
      }
    }
  }
}

#if _WIN32
#pragma endregion
#elif __unix__
#pragma endregion }
#endif

