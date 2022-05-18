
//
#include "Preprocess_KernelFunc.cuh"
#include <float.h>
#include <math.h>

//
__global__ void generate_float_type_depth_KernelFunc(
    const RawDepthType *raw_depth, Sensor_params sensor_params,
    My_Type::Vector2i raw_depth_size, float *float_type_depth) {
  //
  bool is_valid_pixel = true;

  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  // Validate coordinate
  if (u >= raw_depth_size.width || v >= raw_depth_size.height)
    is_valid_pixel = false;

  //
  float float_type_depth_value = 0.0f;
  //
  if (is_valid_pixel) {
    //
    int raw_image_index = u + v * raw_depth_size.width;
    float_type_depth_value =
        (float)raw_depth[raw_image_index] / sensor_params.sensor_scale;
    if (float_type_depth_value < sensor_params.min_range)
      float_type_depth_value = 0.0f;
    if (float_type_depth_value > sensor_params.max_range)
      float_type_depth_value = 0.0f;
  }

  // Save to aligned point image
  int aligned_image_width = blockDim.x * gridDim.x;
  int aligned_image_index = u + v * aligned_image_width;
  float_type_depth[aligned_image_index] = float_type_depth_value;
}
//
void generate_float_type_depth_CUDA(dim3 block_rect, dim3 thread_rect,
                                    const RawDepthType *raw_depth,
                                    Sensor_params sensor_params,
                                    My_Type::Vector2i raw_depth_size,
                                    float *float_type_depth) {

  generate_float_type_depth_KernelFunc<<<block_rect, thread_rect>>>(
      raw_depth, sensor_params, raw_depth_size, float_type_depth);
}

//
__global__ void bilateral_filter_3x3_KernelFunc(const float *src_depth,
                                                float dst_depth) {
  //
  // bool is_valid_pixel = true;
  //// Pixel coordinate
  // int u, v;
  // u = threadIdx.x + blockDim.x * blockIdx.x;
  // v = threadIdx.y + blockDim.y * blockIdx.y;
}
//
void bilateral_filter_3x3_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float dst_depth) {}

//
#define RADIUS_5x5 2
#define SIGMA_L_5x5 1.2232f
//
__global__ void bilateral_filter_5x5_KernelFunc(const float *src_depth,
                                                float *dst_depth) {
  //
  bool is_valid_pixel = true;
  // Image size
  int image_width = blockDim.x * gridDim.x;
  int image_height = blockDim.y * gridDim.y;
  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  int image_index = u + v * image_width;

  // Validate pixel coordinate
  if (u < RADIUS_5x5 || u >= (image_width - RADIUS_5x5))
    is_valid_pixel = false;
  if (v < RADIUS_5x5 || v >= (image_height - RADIUS_5x5))
    is_valid_pixel = false;

  //
  float depth_value = src_depth[image_index];
  if (depth_value == 0.0f)
    is_valid_pixel = false;

  //
  float filtered_depth_value = 0.0f;
  //
  if (is_valid_pixel) {
    float weight_sum = 0.0f;
    // float sigma_z = 1.0f / (0.0012f + 0.0019f * (depth_value - 0.4f) *
    // (depth_value - 0.4f)
    //						+ 0.0001f / sqrtf(depth_value) *
    //0.25f);
    float sigma_z = 1.0f / (0.01f * depth_value);

    //
    for (int i = -2; i <= 2; i++)
      for (int j = -2; j <= 2; j++) {
        float temp_z = src_depth[image_index + i * image_width + j];
        if (temp_z == 0)
          continue;
        //
        float diff_z = temp_z - depth_value;
        //
        float weight =
            expf(-0.5 * ((abs(i) + abs(j)) * SIGMA_L_5x5 * SIGMA_L_5x5 +
                         diff_z * diff_z * sigma_z * sigma_z));
        weight_sum += weight;
        filtered_depth_value += weight * temp_z;
      }

    //
    filtered_depth_value /= weight_sum;
  }

  // Store filtered depth value
  dst_depth[image_index] = filtered_depth_value;
}
//
void bilateral_filter_5x5_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float *dst_depth) {
  bilateral_filter_5x5_KernelFunc<<<block_rect, thread_rect>>>(src_depth,
                                                               dst_depth);
}

//
__global__ void
generate_intensity_image_KernelFunc(const RawColorType *raw_color,
                                    My_Type::Vector2i raw_color_size,
                                    float *aligned_intensity_image) {
  //
  bool is_valid_pixel = true;

  // Pixel coordinate
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  // Validate coordinate
  if (u >= raw_color_size.width || v >= raw_color_size.height)
    is_valid_pixel = false;

  //
  float gray_intensity = 0.0;
  int image_index = u + v * raw_color_size.width;
  if (is_valid_pixel) {
    RawColorType pixel_color = raw_color[image_index];
    // Convert RGB to intensity
    gray_intensity =
        (0.299f * (float)pixel_color.r + 0.587f * (float)pixel_color.g +
         0.114f * (float)pixel_color.b) /
        255.f;
  }
  aligned_intensity_image[image_index] = gray_intensity;
}
//
void generate_intensity_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                   const RawColorType *raw_color,
                                   My_Type::Vector2i raw_color_size,
                                   float *aligned_intensity_image) {
  generate_intensity_image_KernelFunc<<<block_rect, thread_rect>>>(
      raw_color, raw_color_size, aligned_intensity_image);
}

//
#define GRADIENT_STEP 1
__global__ void
generate_gradient_image_KernelFunc(const float *aligned_intensity_image,
                                   My_Type::Vector2f *gradient_image) {
  bool is_valid_pixel = true;
  // Pixel coordinate
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  int v = threadIdx.y + blockDim.y * blockIdx.y;
  //
  int image_width = blockDim.x * gridDim.x;
  int image_height = blockDim.y * gridDim.y;
  if (u < GRADIENT_STEP || u >= (image_width - GRADIENT_STEP))
    is_valid_pixel = false;
  if (v < GRADIENT_STEP || v >= (image_height - GRADIENT_STEP))
    is_valid_pixel = false;
  int image_index = u + v * image_width;

  float gray_intensity[9];
  if (is_valid_pixel) {
    // unroll
    gray_intensity[0] =
        aligned_intensity_image[image_index - image_width * GRADIENT_STEP -
                                GRADIENT_STEP];
    gray_intensity[1] =
        aligned_intensity_image[image_index - image_width * GRADIENT_STEP];
    gray_intensity[2] =
        aligned_intensity_image[image_index - image_width * GRADIENT_STEP +
                                GRADIENT_STEP];
    gray_intensity[3] = aligned_intensity_image[image_index - GRADIENT_STEP];
    gray_intensity[4] = aligned_intensity_image[image_index];
    gray_intensity[5] = aligned_intensity_image[image_index + GRADIENT_STEP];
    gray_intensity[6] =
        aligned_intensity_image[image_index + image_width * GRADIENT_STEP -
                                GRADIENT_STEP];
    gray_intensity[7] =
        aligned_intensity_image[image_index + image_width * GRADIENT_STEP];
    gray_intensity[8] =
        aligned_intensity_image[image_index + image_width * GRADIENT_STEP +
                                GRADIENT_STEP];
    for (int i = 0; i < 9; i++) {
      if (gray_intensity[i] <= 0) {
        is_valid_pixel = false;
        break;
      }
    }
  }

  My_Type::Vector2f gradient_vec(0.0f);
  if (is_valid_pixel) {
    // Gradient X
    gradient_vec.x = gray_intensity[5] * 0.5 + gray_intensity[2] * 0.25 +
                     gray_intensity[8] * 0.25;
    gradient_vec.x -= gray_intensity[3] * 0.5 + gray_intensity[0] * 0.25 +
                      gray_intensity[6] * 0.25;
    gradient_vec.x /= (float)(GRADIENT_STEP * 2);

    // Gradient Y
    gradient_vec.y = gray_intensity[7] * 0.5 + gray_intensity[6] * 0.25 +
                     gray_intensity[8] * 0.25;
    gradient_vec.y -= gray_intensity[1] * 0.5 + gray_intensity[0] * 0.25 +
                      gray_intensity[2] * 0.25;
    gradient_vec.y /= (float)(GRADIENT_STEP * 2);
  }
  //
  gradient_image[image_index] = gradient_vec;
}
//
void generate_gradient_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                  const float *aligned_intensity_image,
                                  My_Type::Vector2f *gradient_image) {

  generate_gradient_image_KernelFunc<<<block_rect, thread_rect>>>(
      aligned_intensity_image, gradient_image);
}

__global__ void outlier_point_filter_KernelFunc(const float *src_depth,
                                                float *dst_depth) {
  bool enable = false;
  bool is_valid_pixel = true;
  int count = 0;
  // Image size
  int image_width = blockDim.x * gridDim.x;
  int image_height = blockDim.y * gridDim.y;
  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  int image_index = u + v * image_width;

  // Validate pixel coordinate
  if (u < 3 || u >= (image_width - 3))
    is_valid_pixel = false;
  if (v < 3 || v >= (image_height - 3))
    is_valid_pixel = false;

  float z, tmpz, dis;

  z = src_depth[image_index];
  if (z <= 0.001f)
    is_valid_pixel = false;

  if (is_valid_pixel)
    for (int i = -3; i < 3; ++i) {   // 3
      for (int j = -3; j < 3; ++j) { // 3
        if (i == 0 && j == 0)
          continue;
        tmpz = src_depth[(u + j) + (v + i) * image_width];
        if (tmpz <= 0.001f)
          continue;

        dis = tmpz - z;
        dis = dis * dis;
        if (dis <
            0.001f) { // 0.005~0.015//多个相邻点深度接近该点深度，该点不是噪点
          count++;
          if (count >= 3)
            enable = true; // 3
          break;
        }
      }
      if (enable)
        break;
    }

  if (!enable) {
    dst_depth[image_index] = 0.0f;
  } else
    dst_depth[image_index] = z;
}

//
#define RADIUS 3
#define SIGMA_L_5x5 5.2232f
//
__global__ void bilateral_filter_KernelFunc(const float *src_depth,
                                            float *dst_depth) {
  //
  bool is_valid_pixel = true;
  // Image size
  int image_width =
      blockDim.x *
      gridDim.x; // TODO remove the img size computation out of the filter, for
                 // example, input para Vector2d directly
  int image_height = blockDim.y * gridDim.y;
  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  int image_index = u + v * image_width;

  // Validate pixel coordinate
  if (u < RADIUS || u >= (image_width - RADIUS))
    is_valid_pixel = false;
  if (v < RADIUS || v >= (image_height - RADIUS))
    is_valid_pixel = false;

  //
  float depth_value = src_depth[image_index];
  if (depth_value <= 0.001f)
    is_valid_pixel = false;

  //
  float filtered_depth_value = 0.0f;
  //
  if (is_valid_pixel) {
    float weight_sum = 0.0f;
    //		float sigma_z = 1.0f / (0.0012f + 0.0019f * (depth_value - 0.4f) *
    //(depth_value - 0.4f)
    //								+ 0.0001f / sqrtf(depth_value) *
    //0.25f);
    float sigma_z = 35.0f / 1000.0f;

    //
    for (int i = -RADIUS; i <= RADIUS; i++)
      for (int j = -RADIUS; j <= RADIUS; j++) {
        float temp_z = src_depth[image_index + i * image_width + j];
        if (temp_z == 0)
          continue;
        //
        float diff_z = temp_z - depth_value;
        diff_z = diff_z * diff_z;

        if (diff_z > 0.001f)
          continue;
        //			float weight = expf(-0.5 * ((abs(i) + abs(j)) *
        //SIGMA_L_5x5 * SIGMA_L_5x5
        //										+ diff_z *
        //diff_z * sigma_z * sigma_z));
        float weight = exp(-0.5f * diff_z * sigma_z * sigma_z);
        weight_sum += weight;
        filtered_depth_value += weight * temp_z;
      }

    if (weight_sum != 0) {
      filtered_depth_value /= weight_sum;
      dst_depth[image_index] = filtered_depth_value;
    } else
      dst_depth[image_index] = 0.0f;
    //

  } else {
    dst_depth[image_index] = 0.0f;
  }

  // Store filtered depth value
}
//
void outlier_point_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                               const float *src_depth, float *dst_depth) {
  outlier_point_filter_KernelFunc<<<block_rect, thread_rect>>>(src_depth,
                                                               dst_depth);
}

void bilateral_filter_CUDA(dim3 block_rect, dim3 thread_rect,
                           const float *src_depth, float *dst_depth) {
  bilateral_filter_KernelFunc<<<block_rect, thread_rect>>>(src_depth,
                                                           dst_depth);
}

//
__global__ void
generate_aligned_points_image_KernelFunc(const float *filtered_depth,
                                         Sensor_params sensor_params,
                                         My_Type::Vector3f *points_image) {
  //
  bool is_valid_pixel = true;

  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  int aligned_image_width = blockDim.x * gridDim.x;

  //
  int aligned_image_index = u + v * aligned_image_width;
  float depth_value = filtered_depth[aligned_image_index];
  if (depth_value == 0.0f)
    is_valid_pixel = false;

  //
  My_Type::Vector3f point(0.0f, 0.0f, 0.0f);
  if (is_valid_pixel) {
    //
    point.x = depth_value * ((float)u - sensor_params.sensor_cx + 0.5f) /
              sensor_params.sensor_fx;
    point.y = depth_value * ((float)v - sensor_params.sensor_cy + 0.5f) /
              sensor_params.sensor_fy;
    point.z = depth_value;
  }

  // Save to aligned point image
  points_image[aligned_image_index] = point;
}
//
void generate_aligned_points_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                        const float *filtered_depth,
                                        Sensor_params sensor_params,
                                        My_Type::Vector3f *points_image) {
  generate_aligned_points_image_KernelFunc<<<block_rect, thread_rect>>>(
      filtered_depth, sensor_params, points_image);
}

//
__global__ void generate_hierarchy_instensity_image_KernelFunc() {}
//
__global__ void generate_hierarchy_points_image_KernelFunc() {}

// ------------------ Compute normals image ------------------
// To do : Evaluate different normal generators
// To do : filter high gradient neighbor pixels
template <int Normal_Stride>
inline __device__ void
compute_normal_vector(const My_Type::Vector3f *points_image, int &image_width,
                      int &center_index, My_Type::Vector3f &normal_vec) {
  //
  My_Type::Vector3f up_point, down_point;
  My_Type::Vector3f left_point, right_point;
  // Load point coordinate and validation
  up_point = points_image[center_index - Normal_Stride * image_width];
  if (up_point.z <= FLT_EPSILON)
    return;
  down_point = points_image[center_index + Normal_Stride * image_width];
  if (down_point.z <= FLT_EPSILON)
    return;
  left_point = points_image[center_index - Normal_Stride];
  if (left_point.z <= FLT_EPSILON)
    return;
  right_point = points_image[center_index + Normal_Stride];
  if (right_point.z <= FLT_EPSILON)
    return;

  My_Type::Vector3f vec_u2d, vec_l2r;
  // vec_u2d : Vector up_point to down_point
  vec_u2d = down_point - up_point;
  // vec_l2r : Vector left_point to right_point
  vec_l2r = right_point - left_point;

  // Compute normal vector
  normal_vec.x = vec_u2d.y * vec_l2r.z - vec_u2d.z * vec_l2r.y;
  normal_vec.y = vec_u2d.z * vec_l2r.x - vec_u2d.x * vec_l2r.z;
  normal_vec.z = vec_u2d.x * vec_l2r.y - vec_u2d.y * vec_l2r.x;
  // Normalize
  float normal_vec_norm = normal_vec.norm();
  normal_vec /= normal_vec_norm;
}

//
#define NORMAL_STRIDE 2
//
template <int Normal_Stride>
__global__ void
compute_normals_image_KernelFunc(const My_Type::Vector3f *points_image,
                                 My_Type::Vector3f *normals_image) {
  //
  bool is_valid_pixel = true;

  //
  int image_width, image_height;
  image_width = blockDim.x * gridDim.x;
  image_height = blockDim.y * gridDim.y;

  // Pixel coordinate
  int u, v;
  u = threadIdx.x + blockDim.x * blockIdx.x;
  v = threadIdx.y + blockDim.y * blockIdx.y;
  if (u < Normal_Stride || u >= image_width - Normal_Stride)
    is_valid_pixel = false;
  if (v < Normal_Stride || v >= image_height - Normal_Stride)
    is_valid_pixel = false;

  //
  My_Type::Vector3f normal_vec(0, 0, 0);
  int normal_image_index = u + v * image_width;
  if (is_valid_pixel) {
    compute_normal_vector<Normal_Stride>(points_image, image_width,
                                         normal_image_index, normal_vec);
  }
  normals_image[normal_image_index] = normal_vec;
}
//
void compute_normals_image_CUDA(dim3 block_rect, dim3 thread_rect,
                                const My_Type::Vector3f *points_image,
                                My_Type::Vector3f *normals_image) {
  compute_normals_image_KernelFunc<NORMAL_STRIDE>
      <<<block_rect, thread_rect>>>(points_image, normals_image);
}

//
template <typename T>
__global__ void down_sample_hierarchy_layers_KernelFunc(
    const T *src_layer, My_Type::Vector2i src_layer_size, T *dst_layer) {
  //
  bool is_valid_pixel = true;

  //
  int dst_u, dst_v;
  dst_u = threadIdx.x + blockDim.x * blockIdx.x;
  dst_v = threadIdx.y + blockDim.y * blockIdx.y;
  // Memory access validate
  if ((dst_u << 1) >= (src_layer_size.width - 1) ||
      (dst_v << 1) >= (src_layer_size.height - 1))
    is_valid_pixel = false;

  //
  T T_sum = 0.0f;
  int valid_counter = 4;
  if (is_valid_pixel) {
    //
    int src_u, src_v;
    src_u = dst_u << 1 + 0;
    src_v = dst_v << 1 + 0;

    T temp_T = 0;
    // (0, 0)
    temp_T = src_layer[src_u + src_v * src_layer_size.width];
    if (temp_T == 0)
      valid_counter--;
    T_sum += temp_T;
    // (0, 1)
    temp_T = src_layer[(src_u + 1) + src_v * src_layer_size.width];
    if (temp_T == 0)
      valid_counter--;
    T_sum += temp_T;
    // (1, 0)
    temp_T = src_layer[src_u + (src_v + 1) * src_layer_size.width];
    if (temp_T == 0)
      valid_counter--;
    T_sum += temp_T;
    // (1, 1)
    temp_T = src_layer[(src_u + 1) + (src_v + 1) * src_layer_size.width];
    if (temp_T == 0)
      valid_counter--;
    T_sum += temp_T;
  }

  //
  if (valid_counter != 0)
    T_sum /= (float)valid_counter;
  dst_layer[dst_u + dst_v * blockDim.x * gridDim.x] = T_sum;
}
// Specialization T = My_Type::Vector3f (ToDo : copy operator for
// My_Type::VectorX )
template <>
__global__ void down_sample_hierarchy_layers_KernelFunc<My_Type::Vector3f>(
    const My_Type::Vector3f *src_layer, My_Type::Vector2i src_layer_size,
    My_Type::Vector3f *dst_layer) {
  //
  bool is_valid_pixel = true;

  //
  int dst_u, dst_v;
  dst_u = threadIdx.x + blockDim.x * blockIdx.x;
  dst_v = threadIdx.y + blockDim.y * blockIdx.y;
  // Memory access validate
  if ((dst_u << 1) >= (src_layer_size.width - 1) ||
      (dst_v << 1) >= (src_layer_size.height - 1))
    is_valid_pixel = false;

  //
  My_Type::Vector3f vec_sum(0.0f);
  int valid_counter = 4;
  if (is_valid_pixel) {
    //
    int src_u, src_v;
    src_u = dst_u << 1;
    src_v = dst_v << 1;

    My_Type::Vector3f temp_vec(0.0f);
    // (0, 0)
    temp_vec = src_layer[src_u + src_v * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (0, 1)
    temp_vec = src_layer[(src_u + 1) + src_v * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (1, 0)
    temp_vec = src_layer[src_u + (src_v + 1) * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (1, 1)
    temp_vec = src_layer[(src_u + 1) + (src_v + 1) * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
  }

  //
  if (valid_counter != 0)
    vec_sum /= (float)valid_counter;
  dst_layer[dst_u + dst_v * blockDim.x * gridDim.x] = vec_sum;
}
//
// Specialization T = My_Type::Vector3f (ToDo : copy operator for
// My_Type::VectorX )
template <>
__global__ void down_sample_hierarchy_layers_KernelFunc<My_Type::Vector2f>(
    const My_Type::Vector2f *src_layer, My_Type::Vector2i src_layer_size,
    My_Type::Vector2f *dst_layer) {
  //
  bool is_valid_pixel = true;

  //
  int dst_u, dst_v;
  dst_u = threadIdx.x + blockDim.x * blockIdx.x;
  dst_v = threadIdx.y + blockDim.y * blockIdx.y;
  // Memory access validate
  if ((dst_u << 1) >= (src_layer_size.width - 1) ||
      (dst_v << 1) >= (src_layer_size.height - 1))
    is_valid_pixel = false;

  //
  My_Type::Vector2f vec_sum(0.0f);
  int valid_counter = 4;
  if (is_valid_pixel) {
    //
    int src_u, src_v;
    src_u = dst_u << 1;
    src_v = dst_v << 1;

    My_Type::Vector2f temp_vec(0.0f);
    // (0, 0)
    temp_vec = src_layer[src_u + src_v * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (0, 1)
    temp_vec = src_layer[(src_u + 1) + src_v * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (1, 0)
    temp_vec = src_layer[src_u + (src_v + 1) * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
    // (1, 1)
    temp_vec = src_layer[(src_u + 1) + (src_v + 1) * src_layer_size.width];
    if (temp_vec == 0)
      valid_counter--;
    vec_sum += temp_vec;
  }

  //
  if (valid_counter != 0)
    vec_sum /= (float)valid_counter;
  dst_layer[dst_u + dst_v * blockDim.x * gridDim.x] = vec_sum;
}
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector2f *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       My_Type::Vector2f *dst_layer) {
  down_sample_hierarchy_layers_KernelFunc<<<block_rect, thread_rect>>>(
      src_layer, src_layer_size, dst_layer);
}
//
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const My_Type::Vector3f *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       My_Type::Vector3f *dst_layer) {
  down_sample_hierarchy_layers_KernelFunc<<<block_rect, thread_rect>>>(
      src_layer, src_layer_size, dst_layer);
}
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const float *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       float *dst_layer) {
  down_sample_hierarchy_layers_KernelFunc<<<block_rect, thread_rect>>>(
      src_layer, src_layer_size, dst_layer);
}
template <typename T>
void down_sample_hierarchy_layers_CUDA(dim3 block_rect, dim3 thread_rect,
                                       const T *src_layer,
                                       My_Type::Vector2i src_layer_size,
                                       T *dst_layer) {
  down_sample_hierarchy_layers_KernelFunc<T>
      <<<block_rect, thread_rect>>>(src_layer, src_layer_size, dst_layer);
}

// 4
//
// void instantiation_down_sample_hierarchy_layers_CUDA()
//{
//	dim3 block_rect, thread_rect;
//	My_Type::Vector2i temp_vec;
//
//	down_sample_hierarchy_layers_CUDA<float>(block_rect, thread_rect, NULL,
//temp_vec, NULL);
//	down_sample_hierarchy_layers_CUDA<My_Type::Vector2f>(block_rect,
//thread_rect, NULL, temp_vec, NULL);
//	down_sample_hierarchy_layers_CUDA<My_Type::Vector3f>(block_rect,
//thread_rect, NULL, temp_vec, NULL);
//}

/*!
        NOTE: Geometry laplacian(mesh) is not better than trival CV laplacian
*/
//
#define LAPLACIAN_STRIDE 2
//
template <int Laplacian_Stride>
__global__ void
compute_points_laplacian_KernelFunc(const My_Type::Vector3f *points_image,
                                    float *laplacian_image) {}
//
void compute_points_laplacian_CUDA(dim3 block_rect, dim3 thread_rect,
                                   const My_Type::Vector3f *points_image,
                                   float *laplacian_image) {
  compute_points_laplacian_KernelFunc<LAPLACIAN_STRIDE>
      <<<block_rect, thread_rect>>>(points_image, laplacian_image);
}
