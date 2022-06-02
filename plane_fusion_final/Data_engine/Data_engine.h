/**
 *  @file Data_engine.h
 *  @brief Data engine to load data and to set the estimated pose storage path.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include <cstdio>
#include <iostream>

#include "Data_engine/Data_loader.h"
#include "Data_engine/Data_writer.h"
using namespace std;

/**
 * @brief This class will be modified as virtul class!
 */
class Data_engine : public Data_loader, public Data_writer {
 public:
  /** @brief Default constructor. */
  Data_engine();
  /** @brief Default destructor. */
  ~Data_engine();

  /**
   * @brief Initialize the imaga_loader and estimated pose output folder.
   * @param image_loader_ptr Assigned value to image_loader.
   * @param output_folder Assigned value to estimated pose output folder.
   * @param _is_ICL_NUIM_dataset Flag whether ICL dataset.
   */
  void init(Image_loader *image_loader_ptr, string output_folder,
            bool _is_ICL_NUIM_dataset = false);
};
