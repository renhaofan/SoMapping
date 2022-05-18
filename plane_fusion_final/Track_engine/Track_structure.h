#pragma once

//
typedef struct struct_Accumulate_result {
  // Hessian  Nabla
  float hessian_upper[21];
  float nabla[6];
  //
  float energy;
  //
  int number_of_pairs;
} Accumulate_result;
