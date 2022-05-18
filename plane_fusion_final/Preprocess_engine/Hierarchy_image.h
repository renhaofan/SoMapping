#pragma once

//
#include <iostream>
//
#include "OurLib/My_matrix.h"
#include "OurLib/my_math_functions.h"

// Type defination of raw data
typedef My_Type::Vector3uc RawColorType;
typedef unsigned short RawDepthType;

//
#define MAX_LAYER_NUMBER 8

//!
/*!

*/
template <typename T> class Hierarchy_image {
public:
  //! Number of layers
  int number_of_layers;
  //! Image size of layer-0
  My_Type::Vector2i size[MAX_LAYER_NUMBER];
  //!
  T *data_ptrs[MAX_LAYER_NUMBER];

  //! Constructor/Destructor
  Hierarchy_image() {}
  ~Hierarchy_image() {}

  //! Initialize parameters
  void init_parameters(My_Type::Vector2i image_size, int alignment_size,
                       int layer_number) {
    //
    this->number_of_layers = layer_number;

    //
    int width, height;
    width = image_size.width;
    height = image_size.height;
    for (size_t layer_id = 0; layer_id < layer_number; layer_id++) {
      //
      this->size[layer_id].width = ceil_by_stride(width, alignment_size);
      this->size[layer_id].height = ceil_by_stride(height, alignment_size);
      //
      width /= 2;
      height /= 2;

      //
      this->data_ptrs[layer_id] = nullptr;
    }
  }
};
