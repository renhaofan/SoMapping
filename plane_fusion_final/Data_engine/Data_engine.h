#pragma once

//
#include "Data_engine/Data_loader.h"
#include "Data_engine/Data_writer.h"

//
#include <cstdio>
#include <iostream>
using namespace std;

//! This class will be modified as virtul class!
/*!

*/
class Data_engine : public Data_loader, public Data_writer {
public:
  //! Default constructor/destructor
  Data_engine();
  ~Data_engine();

  //! Initiation
  void init(Image_loader *image_loader_ptr, string output_folder,
            bool _is_ICL_NUIM_dataset = false);
};
