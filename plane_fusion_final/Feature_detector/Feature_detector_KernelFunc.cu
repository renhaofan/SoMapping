
#include "Feature_detector_KernelFunc.cuh"

__global__ void convert_intensity_to_cvmat_KernelFunc(
    const float *src_intensity, My_Type::Vector2i image_size,
    My_Type::Vector2f intensity_range, unsigned char *dst_intensity) {
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  if (u >= image_size.width)
    return;
  if (v >= image_size.height)
    return;
  int index = v * image_size.width + u;
  // Load intensity value
  float intensity_value_f = src_intensity[index];
  // Normalization
  intensity_value_f = (intensity_value_f - intensity_range.x) /
                      (intensity_range.y - intensity_range.x);
  // Scale to [0, 255]
  intensity_value_f = intensity_value_f * 256.0f;
  int intensity_value_i = (int)intensity_value_f;
  if (intensity_value_i < 0)
    intensity_value_i = 0;
  if (intensity_value_i > 255)
    intensity_value_i = 255;

  //
  dst_intensity[index] = (unsigned char)intensity_value_i;
  // dst_intensity[index] = (unsigned char)0x05 * blockIdx.x * blockIdx.y;
}
//
void convert_intensity_to_cvmat_KernelFunc(dim3 block_rect, dim3 thread_rect,
                                           const float *src_intensity,
                                           My_Type::Vector2i image_size,
                                           My_Type::Vector2f intensity_range,
                                           unsigned char *dst_intensity) {

  convert_intensity_to_cvmat_KernelFunc<<<block_rect, thread_rect>>>(
      src_intensity, image_size, intensity_range, dst_intensity);
}
