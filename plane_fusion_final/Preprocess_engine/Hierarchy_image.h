/**
 *  @file Hierarchy_image.h
 *  @brief Construct hierarchy image with alignment.
 *  @details Only pre-compute image size per layer, not allocate memory. Layer
 *           size not simply half of last layers, but with alignment after half
 *           size operation.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-5-21
 */

#pragma once
#include <iostream>
// extract MAX_LAYER_NUMBER
#include "Config.h"
#include "Log.h"
#include "OurLib/My_matrix.h"
#include "OurLib/my_math_functions.h"

/** @brief Color image raw data type Vector3uc(uchar). */
typedef My_Type::Vector3uc RawColorType;
/** @brief Depth image raw data type unsigned short. */
typedef unsigned short RawDepthType;

template <typename T>
class Hierarchy_image {
 public:
  /** @brief Number of layers. */
  int number_of_layers;
  /** @brief Image size of layers. */
  My_Type::Vector2i size[MAX_LAYER_NUMBER];

  /** @brief Data pointer for different image layer. */
  T *data_ptrs[MAX_LAYER_NUMBER];

  /** @brief Default constructor. */
  Hierarchy_image() {}
  /** @brief Default destructor. */
  ~Hierarchy_image() {}

  /**
   * @brief Pre-compute hierarchy image size per layer.
   * @param image_size Layer-0 image size.
   * @param alignment_size Base size, image size in every layer is integral
   *        mulitiple of alignment_size.
   * @param layer_number The number of image layers.
   * @warning Image size per layer is not simplely the half of last layer,
   * because of alignment_size.
   * @exception Aligned layer_number larger than  macro MAX_LAYER_NUMBER.
   */
  void init_parameters(My_Type::Vector2i image_size, int alignment_size,
                       int layer_number) {
    if (layer_number > MAX_LAYER_NUMBER) {
#ifdef LOGGING
      LOG_FATAL("Pyramid layer number too large");
      Log::shutdown();
#endif
      fprintf(stderr,
              "File %s, Line %d, Function %s(), layer number too large.\n",
              __FILE__, __LINE__, __FUNCTION__);
      throw("layer number too large!");
    }
    this->number_of_layers = layer_number;

    int width, height;
    width = image_size.width;
    height = image_size.height;

    for (int layer_id = 0; layer_id < layer_number; layer_id++) {
      this->size[layer_id].width = ceil_by_stride(width, alignment_size);
      this->size[layer_id].height = ceil_by_stride(height, alignment_size);
      width /= 2;
      height /= 2;
      this->data_ptrs[layer_id] = nullptr;
    }
  }
};
