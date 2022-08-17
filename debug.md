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

如果使用gt slam_system 会打印出这样，到时候定位一下，下面三行在哪里

Triangle mesh size : 320.618	 MB
Voxel array size :	 92.023	 MB
Voxel block number : 23558	 blocks

---



# Program entry:

Three steps in program entry, function main()

```
image_loader_ptr = new Offline_image_loader(cal, dir, dm);
 
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

```c++
Main_engine::instance()->init(argc, argv, image_loader_ptr);
	
    // environment initializer
	this->environment_initializer = new Environment_Initializer(true);
	this->environment_initializer->init_environment(argc, argv);
    
    // data engine
    this->data_engine = new Data_engine();
    this->data_engine->init(image_loader_ptr, output_folder, _is_ICL_NUIM_dataset);

    // slam system initializer
    this->SLAM_system_ptr = new Submap_SLAM_system();  (选择什么SLAM, 7个部分)
    this->SLAM_system_ptr->init(this->data_engine);
		  this->init_parameters(data_engine)
			SLAM_system_settings::instance() set aligned image size
            Create preprocess engine
            Create plane_detector(可以选择用什么方式检测平面)
            Create feature detector
          this->init_modules();
			Init map_engine
            Init track_engine
            Init sub_map
            Init mesh generator

    // render engine
	this->render_engine_ptr = new Render_engine();

    // UI engine initializer
    UI_engine::instance()->init(argc, argv, this->data_engine,
                              this->SLAM_system_ptr, this->render_engine_ptr);
```

```
Main_engine::instance()->run();
    UI_engine::instance()->run() {glutMainLoop()}
```

