/**
 *  Copyright (C) All rights reserved.
 *  @file main_code.cpp
 *  @brief program entry.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 *  @fn void print_argvs(int argc, char** argv);
 *  @fn int main(int argc, char** argv);
 */

#include <cstdio>
#include <iostream>

#include "Log.h"
#include "Main_engine/Main_engine.h"
using namespace std;

/**
 * @brief print arguments.
 * @param argc the number of arguments.
 * @param argv concents of arguments.
 */
void print_argvs(int argc, char **argv);

int main(int argc, char **argv) {
#ifdef LOGGING
  Log::init(&argc, &argv);
  LOG_INFO("<------ Initialising video sequence ...");
#endif

  print_argvs(argc, argv);

  Image_loader *image_loader_ptr = nullptr;

  if (argc == 2) {
    // online image loader
#ifdef LOGGING
    LOG_INFO("Online image loader.");
#endif
  } else if (argc == 4) {
    // offline image loader
    // cal calibration file
    // dir sequence dir
    // dm dataset mode
    string cal = string(argv[1]);
    string dir = string(argv[2]);
    int dm = std::atoi(argv[3]);
    image_loader_ptr = new Offline_image_loader(cal, dir, dm);
#ifdef LOGGING
    LOG_INFO("Offline image loader mode");
    LOG_INFO("Dataset dir: " + dir);
    switch (dm) {
      case 0:
        LOG_INFO("Dataset name: ICL");
        break;
      case 1:
        LOG_INFO("Dataset name: TUM");
        break;
      case 2:
        LOG_INFO("Dataset name: MyZR300");
        break;
      case 3:
        LOG_INFO("Dataset name: MyD435i");
        break;
      case 4:
        LOG_INFO("Dataset name: MyAzureKinect");
        break;
      default:
        LOG_FATAL("Invalid dataset option");
        Log::shutdown();
        break;
    }
#endif
  } else {
    printf(
        "Please input 'calibration file' for online input,\n or "
        "'calibration file, sequence dir, dataset "
        "mode(ICL:0,TUM:1,MyZR300:2,MyD435i:3,MyAzureKinect:4)' for offline "
        "input.\n");

#ifdef LOGGING
    LOG_ERROR("Invalid argc and argv, check that.");
    Log::shutdown();
#endif
    return 0;
  }

#ifdef LOGGING
  LOG_INFO("Initialising sequence finished ------>");
#endif

  string ground_truth_path =
      "/home/steve/dataset/TUM_RGBD_VSLAM/"
      "rgbd_dataset_freiburg1_xyz/groundtruth.txt";

  // Initiation
  Main_engine::instance()->init(argc, argv, image_loader_ptr);
  // Load ground truth file
  Main_engine::instance()->data_engine->load_ground_truth(ground_truth_path,
                                                          true);
#ifdef LOGGING
  LOG_INFO("<------ Run main thread ...");
#endif

  // Run
  Main_engine::instance()->run();

  LOG_FATAL("Abnormal exit");
  Log::shutdown();
  return 0;
}

void print_argvs(int argc, char **argv) {
  printf("argc :\n\t%d\n", argc);
  printf("argvs :\n");
  for (int i = 0; i < argc; i++) printf("\t%s\n", argv[i]);
  printf("\n");
}
