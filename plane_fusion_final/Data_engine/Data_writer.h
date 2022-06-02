/**
 *  @file Data_writer.h
 *  @brief Set up the estimated trajectory storage path.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once

#include <iostream>
using namespace std;
#include "OurLib/Trajectory_node.h"

/**
 * @brief Set up the estimated trajectory storage path.
 */
class Data_writer {
 public:
  /** @brief Flag whether ICL dataset, transform left-hand to right-hand. */
  bool is_ICL_NUIM_dataset = false;

  /** @brief Folder path name, store the estimated trajectory. */
  string output_folder;

  /** @brief Default constructor. */
  Data_writer();
  /** @brief Default destructor. */
  ~Data_writer();

  /**
   * @brief Set estimated trajectory storage path.
   * @param output_folder Estimated poses storage path.
   * @param _is_ICL_NUIM_dataset Whether ICL dataset.
   */
  void init(string output_folder, bool _is_ICL_NUIM_dataset = false);

  /**
   * @brief Save estimated trajectory and shutdown logging.
   * @param estimated_trajectory Estimated trajectory to be saved.
   */
  void save_trajectory(const Trajectory &estimated_trajectory);

 protected:
};
