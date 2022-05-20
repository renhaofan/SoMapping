//! Main file. The entry of this project.
/*!

*/

// C/C++ IO
#include <cstdio>
#include <iostream>
// Main engine
#include "Main_engine/Main_engine.h"
// Namespace
using namespace std;

//! Print argvs
void print_argvs(int, char **);
//!
Image_loader *create_image_loader(string color_folder = "",
                                  string depth_folder = "");

//! Main function.
int main(int argc, char **argv) {
  //
  print_argvs(argc, argv);

  printf("initialising ...\n");
  Image_loader *image_loader_ptr = nullptr;

  if (argc == 2) {
    // online image loader
  } else if (argc == 4) {
    // offline image loader
    string cal = string(argv[1]);
    string dir = string(argv[2]);
    int dm = std::atoi(argv[3]);
    image_loader_ptr = new Offline_image_loader(cal, dir, dm);
  } else {
    printf(
        "Please input 'calibration file' for online input,\n or "
        "'calibration file, sequence dir, dataset "
        "mode(ICL:0,TUM:1,MyZR300:2,MyD435i:3,MyAzureKinect:4)' for offline "
        "input.\n");
    return 0;
  }

  // DBoW3 vocabulary path
  //	const string vocabulary_path =
  //"C:/My_SLAM_project/DBoW_result/vocabulary_orb/voc_L3K5_T500N300.yml.gz";

  //
  //	// fx = fy = 480.0f !
  //    string ground_truth_path = "/home/zhuzunjie/Data/ICL/or2/groundtruth.txt";
  //	string color_folder = "/home/zhuzunjie/Data/ICL/or2/rgb/";
  //	string depth_folder = "/home/zhuzunjie/Data/ICL/or2/depth/";
  //
  //
  //	// Create Image_loader
  //	Image_loader * image_loader_ptr = create_image_loader(color_folder,
  // depth_folder);
  //
  string ground_truth_path = "/home/steve/dataset/TUM_RGBD_VSLAM/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
  // string ground_truth_path = "/home/steve/dataset/scannet0427_00/";
  // Initiation
  Main_engine::instance()->init(argc, argv, image_loader_ptr);
  // Load ground truth file
  Main_engine::instance()->data_engine->load_ground_truth(ground_truth_path,
                                                          true);

  // Run
  Main_engine::instance()->run();

  // Image_loader Test
  if (false) {
    double timestamp;
    cv::Mat color_mat, depth_mat;
    while (Main_engine::instance()->data_engine->load_next_frame(
        timestamp, color_mat, depth_mat, true)) {
      cv::waitKey(10);
    }
  }

  return 0;
}

//! Print argvs
void print_argvs(int argc, char **argv) {
  printf("argc :\n\t%d\n", argc);
  printf("argvs :\n");
  for (int i = 0; i < argc; i++) printf("\t%s\n", argv[i]);

  printf("\n");
}

// Image_loader * create_image_loader(string color_folder, string depth_folder)
//{
//	Image_loader * image_loader_ptr;
//
//	//
//	image_loader_ptr = new Offline_image_loader(color_folder, depth_folder);
//
//	return image_loader_ptr;
//}
