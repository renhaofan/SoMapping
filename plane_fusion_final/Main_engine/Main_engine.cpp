#include "Main_engine.h"

//! The pointer to this static object.
Main_engine *Main_engine::instance_ptr = nullptr;

// Default constructor/destructor
Main_engine::Main_engine() {}
Main_engine::~Main_engine() {
  delete this->environment_initializer;
  delete this->data_engine;
  delete this->SLAM_system_ptr;
  delete this->render_engine_ptr;
}

void Main_engine::init(int argc, char **argv, Image_loader *image_loader_ptr,
                       string output_folder, bool _is_ICL_NUIM_dataset) {
  // <---------- Environment Initializer ---------->
  this->environment_initializer = new Environment_Initializer();
  this->data_engine = new Data_engine();

  // Environment initialization
  this->environment_initializer->init_environment(argc, argv);

  // Data_engine initialization
  this->data_engine->init(image_loader_ptr, output_folder,
                          _is_ICL_NUIM_dataset);

  // <---------- SLAM System Initializer ---------->
  // this->SLAM_system_ptr = new Ground_truth_SLAM_system();
  // this->SLAM_system_ptr = new Basic_voxel_SLAM_system();
  this->SLAM_system_ptr = new Submap_SLAM_system();

  // Create Render engine
  // this->render_engine_ptr->init();
  this->render_engine_ptr = new Render_engine();

  // SLAM_system regiester data_engine
  this->SLAM_system_ptr->init(this->data_engine);

  // UI_engine initialization
  UI_engine::instance()->init(argc, argv, this->data_engine,
                              this->SLAM_system_ptr, this->render_engine_ptr);
}

void Main_engine::run() { UI_engine::instance()->run(); }
