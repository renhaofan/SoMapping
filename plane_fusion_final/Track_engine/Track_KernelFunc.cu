

//
#include "Track_KernelFunc.cuh"
//
#include <float.h>

#include "OurLib/reduction_KernelFunc.cuh"

// Bilinear interpolation
inline __device__ bool interpolate_bilinear(float u, float v,
                                            const float *float_image,
                                            int layer_width, int layer_height,
                                            float &interpolated_float);
//
inline __device__ bool interpolate_bilinear(
    float u, float v, const My_Type::Vector2f *vec2_image, int layer_width,
    int layer_height, My_Type::Vector2f &interpolated_point);
//
inline __device__ bool interpolate_bilinear(
    float u, float v, const My_Type::Vector3f *vec3_image, int layer_width,
    int layer_height, My_Type::Vector3f &interpolated_point,
    float continuous_threshold = FLT_MAX);

// Point-Plane residual
__global__ void compute_points_residual_KernelFunc(
    const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, const float *points_weight,
    Sensor_params sensor_param, My_Type::Matrix44f incremental_pose,
    int layer_id, Accumulate_result *accumulate_result) {
  bool valid_point = true;

  // --------- Load data
  // Current layer size
  int layer_width, layer_height;
  layer_width = gridDim.x * blockDim.x;
  layer_height = gridDim.y * blockDim.y;

  //
  int u_current, v_current;
  u_current = threadIdx.x + blockIdx.x * blockDim.x;
  v_current = threadIdx.y + blockIdx.y * blockDim.y;
  int current_index = u_current + v_current * layer_width;
  // Validate memory access
  if (u_current < 0 || u_current >= layer_width || v_current < 0 ||
      v_current >= layer_height)
    valid_point = false;

  //
  My_Type::Vector3f current_point, model_point, trans_point;
  // Read current point position
  if (valid_point) {
    current_point = current_points[current_index];
    if (current_point.z == 0.0f) valid_point = false;
  }

  float u_model, v_model;
  if (valid_point) {
    // Transfer to iterated position
    trans_point.x = incremental_pose.m00 * current_point.x +
                    incremental_pose.m01 * current_point.y +
                    incremental_pose.m02 * current_point.z +
                    incremental_pose.m03;
    trans_point.y = incremental_pose.m10 * current_point.x +
                    incremental_pose.m11 * current_point.y +
                    incremental_pose.m12 * current_point.z +
                    incremental_pose.m13;
    trans_point.z = incremental_pose.m20 * current_point.x +
                    incremental_pose.m21 * current_point.y +
                    incremental_pose.m22 * current_point.z +
                    incremental_pose.m23;

    // Project current point to (u, v) coordinate (Find correspondence points)
    if (layer_id == 0) {
      u_model = roundf(trans_point.x / trans_point.z * sensor_param.sensor_fx +
                       sensor_param.sensor_cx - 0.5f);
      v_model = roundf(trans_point.y / trans_point.z * sensor_param.sensor_fy +
                       sensor_param.sensor_cy - 0.5f);
    } else {
      u_model = roundf(trans_point.x / trans_point.z * sensor_param.sensor_fx +
                       sensor_param.sensor_cx);
      v_model = roundf(trans_point.y / trans_point.z * sensor_param.sensor_fy +
                       sensor_param.sensor_cy);
      u_model = u_model / (float)(1 << layer_id);
      v_model = v_model / (float)(1 << layer_id);
    }
    // Validate memory access
    if (u_model <= FLT_EPSILON ||
        u_model >= (float)(layer_width - (1 << layer_id)) ||
        v_model <= FLT_EPSILON ||
        v_model >= (float)(layer_height - (1 << layer_id)))
      valid_point = false;
  }

  // Read model points information
  My_Type::Vector3f current_normal, model_normal, trans_normal;
  if (valid_point) {
    // Read normals
    current_normal = current_normals[current_index];
    trans_normal.x = incremental_pose.m00 * current_normal.x +
                     incremental_pose.m01 * current_normal.y +
                     incremental_pose.m02 * current_normal.z;
    trans_normal.y = incremental_pose.m10 * current_normal.x +
                     incremental_pose.m11 * current_normal.y +
                     incremental_pose.m12 * current_normal.z;
    trans_normal.z = incremental_pose.m20 * current_normal.x +
                     incremental_pose.m21 * current_normal.y +
                     incremental_pose.m22 * current_normal.z;

    // Interpolate Read model point/normal
#if (1) /* IMPORTANT : use interpolated points */
    float noise_radius = trans_point.z * sensor_param.sensor_noise_ratio;
    valid_point =
        interpolate_bilinear(u_model, v_model, model_points, layer_width,
                             layer_height, model_point, noise_radius);
    // valid_point = interpolate_bilinear(u_model, v_model, model_points,
    // layer_width, layer_height, model_point, 0.01);
    float normal_continuous_threshold = 0.1;
    valid_point = interpolate_bilinear(u_model, v_model, model_normals,
                                       layer_width, layer_height, model_normal,
                                       normal_continuous_threshold);
#else
    int model_index = (int)u_model + ((int)v_model) * layer_width;
    model_point = model_points[model_index];
    model_normal = model_normals[model_index];
#endif
  }

  // Validate normal direction
  float inner_product = trans_normal.dot(model_normal);
  // if (inner_product < 0.8f)	valid_point = false;

  // --------- Compute residual
  My_Type::Vector3f diff_vec;
  float distance = 0.0f;
  if (valid_point) {
    //
    diff_vec = trans_point - model_point;
    distance = fabsf(diff_vec.dot(model_normal));
    // To Do : use huber functor here !
    // if (distance > 0.1f)		valid_point = false;
    if (distance > 0.05f) valid_point = false;
    if (diff_vec.norm() > 0.1f) valid_point = false;
  }

  // Hessian, Nabla
  float A_array[6], b_value, weight = 0.0f;
  if (valid_point) {
#if (1) /* To Do : use precomputed weight here */
    weight = 1;
#endif
    // p_{i} \times n_{i}
    A_array[0] =
        (trans_point.y * model_normal.z - trans_point.z * model_normal.y);
    A_array[1] =
        (trans_point.z * model_normal.x - trans_point.x * model_normal.z);
    A_array[2] =
        (trans_point.x * model_normal.y - trans_point.y * model_normal.x);
    // n_{i}
    A_array[3] = model_normal.x;
    A_array[4] = model_normal.y;
    A_array[5] = model_normal.z;

    // (p_{i} - q_{i}) \cdot n_{i}
    b_value = diff_vec.x * model_normal.x + diff_vec.y * model_normal.y +
              diff_vec.z * model_normal.z;
  } else {
    A_array[0] = 0;
    A_array[3] = 0;
    A_array[1] = 0;
    A_array[4] = 0;
    A_array[2] = 0;
    A_array[5] = 0;
    b_value = 0;
  }
  __syncthreads();

#pragma region(Reduce)
  // Reduce
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  __shared__ float cache_f[256];
  __shared__ int cache_i[256];

  //
  // ATA = Hessain / 2
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++) {
      __syncthreads();
      cache_f[tid] = A_array[i] * A_array[j] * weight;

      //
      block_256_reduce(cache_f, tid);
      //
      if (tid == 0) {
        atomicAdd(&accumulate_result->hessian_upper[(i + 1) * i / 2 + j],
                  cache_f[0]);
      }
    }
  }
  // ATb = Nabla / 2
  for (int i = 0; i < 6; i++) {
    __syncthreads();
    cache_f[tid] = A_array[i] * b_value * weight;

    //
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0) {
      atomicAdd(&accumulate_result->nabla[i], cache_f[0]);
    }
  }

  //
  __syncthreads();
  cache_f[tid] = distance;
  //
  block_256_reduce(cache_f, tid);
  //
  if (tid == 0) {
    atomicAdd(&accumulate_result->energy, cache_f[0]);
  }

  //
  if (valid_point) {
    cache_i[tid] = 1;
  } else {
    cache_i[tid] = 0;
  }
  //
  block_256_reduce(cache_i, tid);
  //
  if (tid == 0) {
    atomicAdd(&accumulate_result->number_of_pairs, cache_i[0]);
  }

#pragma endregion
}
//
void compute_points_residual_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, const float *points_weight,
    Sensor_params sensor_param, My_Type::Matrix44f incremental_pose,
    int layer_id, Accumulate_result *accumulate_result) {
  compute_points_residual_KernelFunc<<<block_rect, thread_rect>>>(
      current_points, model_points, current_normals, model_normals,
      points_weight, sensor_param, incremental_pose, layer_id,
      accumulate_result);
}

//
__global__ void generate_correspondence_lines_KernelFunc(
    const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, Sensor_params sensor_param,
    My_Type::Matrix44f incremental_pose,
    My_Type::Vector3f *correspondence_lines) {
  bool valid_point = true;

  // --------- Load data
  // Current layer size
  int layer_width = gridDim.x * blockDim.x;
  int layer_height = gridDim.y * blockDim.y;

  //
  int u_current, v_current;
  u_current = threadIdx.x + blockIdx.x * blockDim.x;
  v_current = threadIdx.y + blockIdx.y * blockDim.y;
  int current_index = u_current + v_current * layer_width;
  // Validate memory access
  if (u_current < 0 || u_current >= layer_width || v_current < 0 ||
      v_current >= layer_height)
    valid_point = false;

  //
  My_Type::Vector3f current_point, model_point, trans_point;
  // Read current point position
  if (valid_point) {
    current_point = current_points[current_index];
    if (current_point.z == 0.0f) valid_point = false;
  }

  float u_model, v_model;
  if (valid_point) {
    // Transfer to iterated position
    trans_point.x = incremental_pose.m00 * current_point.x +
                    incremental_pose.m01 * current_point.y +
                    incremental_pose.m02 * current_point.z +
                    incremental_pose.m03;
    trans_point.y = incremental_pose.m10 * current_point.x +
                    incremental_pose.m11 * current_point.y +
                    incremental_pose.m12 * current_point.z +
                    incremental_pose.m13;
    trans_point.z = incremental_pose.m20 * current_point.x +
                    incremental_pose.m21 * current_point.y +
                    incremental_pose.m22 * current_point.z +
                    incremental_pose.m23;

    // Project current point to (u, v) coordinate (Find correspondence points)
    u_model = roundf(trans_point.x / trans_point.z * sensor_param.sensor_fx +
                     sensor_param.sensor_cx - 0.5f);
    v_model = roundf(trans_point.y / trans_point.z * sensor_param.sensor_fy +
                     sensor_param.sensor_cy - 0.5f);

    // Validate memory access
    if (u_model <= FLT_EPSILON || u_model >= (float)layer_width ||
        v_model <= FLT_EPSILON || v_model >= (float)layer_height)
      valid_point = false;
  }

  // Read model points information
  My_Type::Vector3f current_normal, model_normal, trans_normal;
  if (valid_point) {
    // Read normals
    current_normal = current_normals[current_index];
    trans_normal.x = incremental_pose.m00 * current_normal.x +
                     incremental_pose.m01 * current_normal.y +
                     incremental_pose.m02 * current_normal.z;
    trans_normal.y = incremental_pose.m10 * current_normal.x +
                     incremental_pose.m11 * current_normal.y +
                     incremental_pose.m12 * current_normal.z;
    trans_normal.z = incremental_pose.m20 * current_normal.x +
                     incremental_pose.m21 * current_normal.y +
                     incremental_pose.m22 * current_normal.z;

    // Interpolate Read model point/normal
#if (1) /* IMPORTANT : use interpolated points */
    float noise_radius = trans_point.z * sensor_param.sensor_noise_ratio;
    valid_point =
        interpolate_bilinear(u_model, v_model, model_points, layer_width,
                             layer_height, model_point, noise_radius);
    float normal_continuous_threshold = 0.1;
    valid_point = interpolate_bilinear(u_model, v_model, model_normals,
                                       layer_width, layer_height, model_normal,
                                       normal_continuous_threshold);
#else
    int model_index = (int)u_model + ((int)v_model) * layer_width;
    model_point = model_points[model_index];
    model_normal = model_normals[model_index];
#endif
  }

  // Validate normal direction
  float inner_product = trans_normal.dot(model_normal);
  // if (inner_product < 0.8f)	valid_point = false;

  // --------- Compute residual
  My_Type::Vector3f diff_vec;
  float distance = 0.0f;
  if (valid_point) {
    //
    diff_vec = trans_point - model_point;
    distance = fabsf(diff_vec.dot(model_normal));
    // To Do : use huber functor here !
    // if (distance > 0.1f)		valid_point = false;
    if (distance > 0.03f) valid_point = false;
    if (diff_vec.norm() > 0.1f) valid_point = false;
  }

  if (valid_point) {
    if (diff_vec.norm() > 0.5) {
      printf("%f, %f, %f\n", model_normal.x, model_normal.y, model_normal.z);
    }
    //
    correspondence_lines[current_index * 2 + 0] = trans_point;
    correspondence_lines[current_index * 2 + 1] = model_point;
  } else {
    correspondence_lines[current_index * 2 + 0] =
        My_Type::Vector3f(0.0, 0.0, 0.0);
    correspondence_lines[current_index * 2 + 1] =
        My_Type::Vector3f(0.0, 0.0, 0.0);
  }
}
//
void generate_correspondence_lines_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const My_Type::Vector3f *model_points,
    const My_Type::Vector3f *current_normals,
    const My_Type::Vector3f *model_normals, Sensor_params sensor_param,
    My_Type::Matrix44f incremental_pose,
    My_Type::Vector3f *correspondence_lines) {
  generate_correspondence_lines_KernelFunc<<<block_rect, thread_rect>>>(
      current_points, model_points, current_normals, model_normals,
      sensor_param, incremental_pose, correspondence_lines);
}

// Photometric residual
__global__ void compute_photometric_residual_KernelFunc(
    const My_Type::Vector3f *current_points, const float *current_intensity,
    const float *model_intensity, const My_Type::Vector2f *model_gradient,
    const float *points_weight, Sensor_params sensor_param,
    My_Type::Matrix44f incremental_pose, int layer_id,
    Accumulate_result *accumulate_result) {
  bool valid_pixel = true;

  // Current layer size
  int layer_width = gridDim.x * blockDim.x;
  int layer_height = gridDim.y * blockDim.y;

  //
  int u_current = threadIdx.x + blockIdx.x * blockDim.x;
  int v_current = threadIdx.y + blockIdx.y * blockDim.y;
  int current_index = u_current + v_current * layer_width;
  // Validate memory access
  if (u_current < 0 || u_current >= layer_width || v_current < 0 ||
      v_current >= layer_height)
    valid_pixel = false;

  //
  My_Type::Vector3f current_point, trans_point;
  if (valid_pixel) {
    current_point = current_points[current_index];
    if (current_point.z == 0.0f) valid_pixel = false;
  }

  float u_model, v_model;
  if (valid_pixel) {
    // Transfer to iterated position
    trans_point.x = incremental_pose.m00 * current_point.x +
                    incremental_pose.m01 * current_point.y +
                    incremental_pose.m02 * current_point.z +
                    incremental_pose.m03;
    trans_point.y = incremental_pose.m10 * current_point.x +
                    incremental_pose.m11 * current_point.y +
                    incremental_pose.m12 * current_point.z +
                    incremental_pose.m13;
    trans_point.z = incremental_pose.m20 * current_point.x +
                    incremental_pose.m21 * current_point.y +
                    incremental_pose.m22 * current_point.z +
                    incremental_pose.m23;

    // Project current point to (u, v) coordinate (Find correspondence points)
    if (layer_id == 0) {
      u_model = roundf(trans_point.x / trans_point.z * sensor_param.sensor_fx +
                       sensor_param.sensor_cx - 0.5f);
      v_model = roundf(trans_point.y / trans_point.z * sensor_param.sensor_fy +
                       sensor_param.sensor_cy - 0.5f);
    } else {
      u_model = roundf(trans_point.x / trans_point.z * sensor_param.sensor_fx +
                       sensor_param.sensor_cx);
      v_model = roundf(trans_point.y / trans_point.z * sensor_param.sensor_fy +
                       sensor_param.sensor_cy);
      u_model = u_model / (float)(1 << layer_id);
      v_model = v_model / (float)(1 << layer_id);
    }
    // Validate memory access
    if (u_model <= FLT_EPSILON ||
        u_model >= (float)(layer_width - (1 << layer_id)) ||
        v_model <= FLT_EPSILON ||
        v_model >= (float)(layer_height - (1 << layer_id)))
      valid_pixel = false;
  }

  //
  float intensity_diff = 0.0f;
  My_Type::Vector2f model_gradient_vec(0.0f);
  if (valid_pixel) {
    float current_pixel_intensity, model_pixel_intensity;
    current_pixel_intensity = current_intensity[current_index];
    valid_pixel |=
        interpolate_bilinear(u_model, v_model, model_intensity, layer_width,
                             layer_height, model_pixel_intensity);
    valid_pixel |=
        interpolate_bilinear(u_model, v_model, model_gradient, layer_width,
                             layer_height, model_gradient_vec);
    if (current_pixel_intensity == 0.0f || model_pixel_intensity == 0.0f)
      valid_pixel = false;
    intensity_diff = model_pixel_intensity - current_pixel_intensity;
    if (fabsf(intensity_diff) < 0.10f || fabsf(intensity_diff) > 0.9f)
      valid_pixel = false;
  }

  //
  float nabla[6];
  if (valid_pixel) {
    float gradient_projection[3];
    gradient_projection[0] = model_gradient_vec.x * sensor_param.sensor_fx;
    gradient_projection[1] = model_gradient_vec.y * sensor_param.sensor_fy;
    gradient_projection[2] = model_gradient_vec.x * sensor_param.sensor_cx +
                             model_gradient_vec.y * sensor_param.sensor_cy;

    //
    nabla[0] = -gradient_projection[1] * current_point.z +
               gradient_projection[2] * current_point.y;
    nabla[1] = +gradient_projection[0] * current_point.z -
               gradient_projection[2] * current_point.x;
    nabla[2] = -gradient_projection[0] * current_point.y +
               gradient_projection[1] * current_point.x;
    nabla[3] = +gradient_projection[0];
    nabla[4] = +gradient_projection[1];
    nabla[5] = +gradient_projection[2];
  } else {
    nabla[0] = 0.0f;
    nabla[1] = 0.0f;
    nabla[2] = 0.0f;
    nabla[3] = 0.0f;
    nabla[4] = 0.0f;
    nabla[5] = 0.0f;
  }

#pragma region(Reduce)
  // Reduce
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  __shared__ float cache_f[256];
  __shared__ int cache_i[256];

  //
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++) {
      __syncthreads();
      cache_f[tid] = nabla[i] * nabla[j];

      //
      block_256_reduce(cache_f, tid);
      //
      if (tid == 0) {
        atomicAdd(&accumulate_result->hessian_upper[(i + 1) * i / 2 + j],
                  cache_f[0]);
      }
    }
  }
  for (int i = 0; i < 6; i++) {
    __syncthreads();
    cache_f[tid] = nabla[i] * intensity_diff;

    //
    block_256_reduce(cache_f, tid);
    //
    if (tid == 0) {
      atomicAdd(&accumulate_result->nabla[i], cache_f[0]);
    }
  }

  //
  __syncthreads();
  cache_f[tid] = intensity_diff;
  //
  block_256_reduce(cache_f, tid);
  //
  if (tid == 0) {
    atomicAdd(&accumulate_result->energy, cache_f[0]);
  }

  //
  if (valid_pixel) {
    cache_i[tid] = 1;
  } else {
    cache_i[tid] = 0;
  }
  //
  block_256_reduce(cache_i, tid);
  //
  if (tid == 0) {
    atomicAdd(&accumulate_result->number_of_pairs, cache_i[0]);
  }

#pragma endregion
}
// Photometric residual
void compute_photometric_residual_CUDA(
    dim3 block_rect, dim3 thread_rect, const My_Type::Vector3f *current_points,
    const float *current_intensity, const float *model_intensity,
    const My_Type::Vector2f *model_gradient, const float *points_weight,
    Sensor_params sensor_param, My_Type::Matrix44f incremental_pose,
    int layer_id, Accumulate_result *accumulate_result) {
  compute_photometric_residual_KernelFunc<<<block_rect, thread_rect>>>(
      current_points, current_intensity, model_intensity, model_gradient,
      points_weight, sensor_param, incremental_pose, layer_id,
      accumulate_result);
}

//
inline __device__ bool interpolate_bilinear(float u, float v,
                                            const float *float_image,
                                            int layer_width, int layer_height,
                                            float &interpolated_float) {
  float coeff_x, coeff_y;
  coeff_x = u - floor(u);
  coeff_y = v - floor(v);

  //
  int base_index = (int)u + (int)v * layer_width;
#if (1) /* for robust */
  if (u < FLT_EPSILON || u >= layer_width - 1.0f || v < FLT_EPSILON ||
      v >= layer_height - 1.0f)
    return false;
#endif

  //
  float point_00, point_01, point_10, point_11;
  point_00 = float_image[base_index];
  if (point_00 == 0.0f) return false;
  point_01 = float_image[base_index + 1];
  if (point_01 == 0.0f) return false;
  point_10 = float_image[base_index + layer_width];
  if (point_10 == 0.0f) return false;
  point_11 = float_image[base_index + layer_width + 1];
  if (point_11 == 0.0f) return false;
  //
  float point_0, point_1;
  point_0 = (1 - coeff_x) * point_00 + coeff_x * point_01;
  point_1 = (1 - coeff_x) * point_10 + coeff_x * point_11;
  //
  interpolated_float = (1 - coeff_y) * point_0 + coeff_y * point_1;

  return true;
}

//
inline __device__ bool interpolate_bilinear(
    float u, float v, const My_Type::Vector2f *vec2_image, int layer_width,
    int layer_height, My_Type::Vector2f &interpolated_point) {
  float coeff_x, coeff_y;
  coeff_x = u - floor(u);
  coeff_y = v - floor(v);

  //
  int base_index = (int)u + (int)v * layer_width;
#if (1) /* for robust */
  if (u < FLT_EPSILON || u >= layer_width - 1.0f || v < FLT_EPSILON ||
      v >= layer_height - 1.0f)
    return false;
#endif

  //
  My_Type::Vector2f point_00, point_01, point_10, point_11;
  point_00 = vec2_image[base_index];
  if (point_00.y == 0.0f) return false;
  point_01 = vec2_image[base_index + 1];
  if (point_01.y == 0.0f) return false;
  point_10 = vec2_image[base_index + layer_width];
  if (point_10.y == 0.0f) return false;
  point_11 = vec2_image[base_index + layer_width + 1];
  if (point_11.y == 0.0f) return false;
  //
  My_Type::Vector2f point_0, point_1;
  point_0 = (1 - coeff_x) * point_00 + coeff_x * point_01;
  point_1 = (1 - coeff_x) * point_10 + coeff_x * point_11;
  //
  interpolated_point = (1 - coeff_y) * point_0 + coeff_y * point_1;

  return true;
}

//
inline __device__ bool interpolate_bilinear(
    float u, float v, const My_Type::Vector3f *vec3_image, int layer_width,
    int layer_height, My_Type::Vector3f &interpolated_point,
    float continuous_threshold) {
  float coeff_x = u - floor(u);
  float coeff_y = v - floor(v);

  //
  int base_index = (int)u + (int)v * layer_width;
#if (1) /* for robust */
  if (u < FLT_EPSILON || u >= layer_width - 1.0f || v < FLT_EPSILON ||
      v >= layer_height - 1.0f)
    return false;
#endif

  //
  My_Type::Vector3f point_00, point_01, point_10, point_11;
  point_00 = vec3_image[base_index];
  if (point_00.z == 0.0f) return false;
  point_01 = vec3_image[base_index + 1];
  if (point_01.z == 0.0f) return false;
  point_10 = vec3_image[base_index + layer_width];
  if (point_10.z == 0.0f) return false;
  point_11 = vec3_image[base_index + layer_width + 1];
  if (point_11.z == 0.0f) return false;
  //
  My_Type::Vector3f point_0, point_1;
  point_0 = (1 - coeff_x) * point_00 + coeff_x * point_01;
  point_1 = (1 - coeff_x) * point_10 + coeff_x * point_11;
  //
  interpolated_point = (1 - coeff_y) * point_0 + coeff_y * point_1;

#define CHECK_COUNTINUOUS
#ifdef CHECK_COUNTINUOUS
  if ((interpolated_point - point_00).norm() > continuous_threshold)
    return false;
  if ((interpolated_point - point_01).norm() > continuous_threshold)
    return false;
  if ((interpolated_point - point_10).norm() > continuous_threshold)
    return false;
  if ((interpolated_point - point_11).norm() > continuous_threshold)
    return false;
#endif

  return true;
}
