```cpp
void UI_engine::render_sub_viewport1() {
  if (this->SLAM_system_ptr->color_mat.empty()) return;

  // wrong
  LOG_WARNING(this->SLAM_system_ptr->color_mat.size().width);
  LOG_WARNING(this->SLAM_system_ptr->color_mat.size().height);

  if (true) {
    drawKeypoints(this->SLAM_system_ptr->color_mat.clone(),
                  this->SLAM_system_ptr->feature_detector->current_keypoints,
                  this->SLAM_system_ptr->color_mat, cv::Scalar(0, 0, 255),
                  cv::DrawMatchesFlags::DEFAULT);
  }

  glViewport(this->main_viewport_width, this->sub_viewport_height,
             this->sub_viewport_width, this->sub_viewport_height);
```

```c++
# 需要添加depth2color align的判断
bool Offline_image_loader::load_next_frame(double &timestamp,
                                           cv::Mat &color_mat,
                                           cv::Mat &depth_mat);                                  
```

```c++
void Offline_image_loader::read_calibration_parameters(string cal) {
    ...
  SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy, 1.0f / scale);
  SLAM_system_settings::instance()->set_extrinsic(d2c, d2i);        
}  


```

```c++
plane_label_mapper[plane_label].y = 365 报错
Mesh_generator_KernelFunc.cu
    
    if (is_planar_voxel) {
        if (plane_label < 0 || plane_label > 127)
            printf("plane_label = %d\n", plane_label);
        if (plane_label_mapper[plane_label].y < 0 ||
            plane_label_mapper[plane_label].y > 127)
            printf("plane_label_mapper[plane_label].y = %d\n",
                   plane_label_mapper[plane_label].y);
        if (plane_label_mapper[plane_label].y > 0)
            plane_label = plane_label_mapper[plane_label].y;

        int plane_color_index = (plane_label * PLANE_COLOR_STEP) % COLOR_NUM;

```

---



# Program entry:

1.  Create image_loader

```
image_loader_ptr = new Offline_image_loader(cal, dir, dm);
```

```c++
  // constructor
  detect_images(associate_dir, color_dir, depth_dir, dm);
    vector<string> image_timestamp_vector
    vector<string> color_path_vector
    vector<string> depth_path_vector
    size_t number_of_frames
  this->read_calibration_parameters(cal);
    SLAM_system_settings set_to_default();
    SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy);
    SLAM_system_settings::instance()->set_intrinsic(fx, fy, cx, cy, 1.0f / scale);
    SLAM_system_settings::instance()->set_extrinsic(d2c, d2i);  (NOT USED????)
    
```
