#pragma once

#include <iostream>
using namespace std;
//!
#include "OurLib/Trajectory_node.h"

//!
/*!

*/
class Data_writer {
 public:
  //!
  bool is_ICL_NUIM_dataset = false;
  //!
  string output_folder;

  //! Default constructor/destructor
  Data_writer();
  ~Data_writer();

  //! Initiate Data_writer
  void init(string output_folder, bool _is_ICL_NUIM_dataset = false);

  //! Save trajectory
  void save_trajectory(const Trajectory &estimated_trajectory);

 protected:
};
