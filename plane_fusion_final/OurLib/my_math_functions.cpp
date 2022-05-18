// My math functions for CPP
//   ：
//   ：2018 6 1  19:33:10
//   ：

//
#include "my_math_functions.h"
// C++
#include "math.h"

// int functions
#pragma region(int functions)
// inline
//
int floor_by_stride(int value, int stride) {
  return ((value - 1) / stride * stride);
}

//
int round_by_stride(int value, int stride) {
  return ((value + (stride >> 1) - 1) / stride * stride);
}

//
int ceil_by_stride(int value, int stride) {
  return (((value - 1) / stride + 1) * stride);
}

#pragma endregion

// float functions
#pragma region(float functions)
//
float floor_by_stride(float value, float stride) {
  return (floorf(value / stride) * stride);
}

//
float round_by_stride(float value, float stride) {
  return (roundf(value / stride) * stride);
}

//
float ceil_by_stride(float value, float stride) {
  return (ceilf(value / stride) * stride);
}
#pragma endregion

// double functions
#pragma region(double functions)
//
double floor_by_stride(double value, double stride) {
  return (floor(value / stride) * stride);
}

//
double round_by_stride(double value, double stride) {
  return (round(value / stride) * stride);
}

//
double ceil_by_stride(double value, double stride) {
  return (ceil(value / stride) * stride);
}
#pragma endregion

// float
int compare_float(const void *a, const void *b) {
  if (*(float *)a > *(float *)b) {
    return -1;
  } else {
    return 1;
  }
}

//   4     float
int compare_float_elm4_by4(const void *a, const void *b) {
  if ((*(float *)a + 3) > *((float *)b + 3)) {
    return -1;
  } else {
    return 1;
  }
}

//   1     float
int compare_float_elm3_by1(const void *a, const void *b) {
  if ((*(float *)a + 0) > *((float *)b + 0)) {
    return -1;
  } else {
    return 1;
  }
}
