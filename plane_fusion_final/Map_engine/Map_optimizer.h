//! Definations of map optimzer
//! Define the Ceres residual functor.

#pragma once

// Ceres
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// My data type
#include "OurLib/My_matrix.h"
#include "OurLib/My_vector.h"

//
#include "Eigen/Dense"

// ----------------------------------------- Ceres optimizer
//! Functor. Only use sparse features to optimze fragment maps' pose.
struct Cost_function_of_feature {
  Eigen::Vector3f _p1, _p2;
  // Assignment constructor
  Cost_function_of_feature(Eigen::Vector3f p1, Eigen::Vector3f p2)
      : _p1(p1), _p2(p2) {}

  //
  template <typename T>
  bool operator()(const T *const extrinsic_1, const T *const extrinsic_2,
                  T *residual) const {
    T p_buffer[3];
    T p_1[3];
    p_buffer[0] = T(_p1.x());
    p_buffer[1] = T(_p1.y());
    p_buffer[2] = T(_p1.z());
    ceres::AngleAxisRotatePoint(extrinsic_1, p_buffer, p_1);
    p_1[0] += extrinsic_1[3];
    p_1[1] += extrinsic_1[4];
    p_1[2] += extrinsic_1[5];

    T p_2[3];
    p_buffer[0] = T(_p2.x());
    p_buffer[1] = T(_p2.y());
    p_buffer[2] = T(_p2.z());
    ceres::AngleAxisRotatePoint(extrinsic_2, p_buffer, p_2);
    p_2[0] += extrinsic_2[3];
    p_2[1] += extrinsic_2[4];
    p_2[2] += extrinsic_2[5];

    double weight = 1.0;

    //
    residual[0] = weight * ceres::abs(p_2[0] - p_1[0]);
    residual[1] = weight * ceres::abs(p_2[1] - p_1[1]);
    residual[2] = weight * ceres::abs(p_2[2] - p_1[2]);

    return true;
  }
};

// TODO
//! Functor. Use adjacent fragment map's plane information to optimize fragment
//! maps' pose.
struct Cost_function_of_adjacent_plane {
  Eigen::Vector3f _n1_F, _n2_F;
  float _d1_F, _d2_F;
  // Assignment constructor
  Cost_function_of_adjacent_plane(Eigen::Vector3f n1_F, Eigen::Vector3f n2_F,
                                  float d1_F, float d2_F)
      : _n1_F(n1_F), _n2_F(n2_F), _d1_F(d1_F), _d2_F(d2_F) {}

  //
  template <typename T>
  bool operator()(const T *const extrinsic_1, const T *const extrinsic_2,
                  T *residual) const {
    //
    double weight = 30.0f;
    // double weight = 1.0;

    T d1_W, d2_W;

    //
    T buffer[3], point_W[3];
    //
    T n1_W[3];
    buffer[0] = T(_n1_F.x());
    buffer[1] = T(_n1_F.y());
    buffer[2] = T(_n1_F.z());
    ceres::AngleAxisRotatePoint(extrinsic_1, buffer, n1_W);
    //
    point_W[0] = -n1_W[0] * (T)_d1_F + extrinsic_1[3];
    point_W[1] = -n1_W[1] * (T)_d1_F + extrinsic_1[4];
    point_W[2] = -n1_W[2] * (T)_d1_F + extrinsic_1[5];
    d1_W = ceres::abs(point_W[0] * n1_W[0] + point_W[1] * n1_W[1] +
                      point_W[2] * n1_W[2]);

    //
    T n2_W[3];
    buffer[0] = T(_n2_F.x());
    buffer[1] = T(_n2_F.y());
    buffer[2] = T(_n2_F.z());
    ceres::AngleAxisRotatePoint(extrinsic_2, buffer, n2_W);
    //
    point_W[0] = -n2_W[0] * (T)_d2_F + extrinsic_2[3];
    point_W[1] = -n2_W[1] * (T)_d2_F + extrinsic_2[4];
    point_W[2] = -n2_W[2] * (T)_d2_F + extrinsic_2[5];
    d2_W = ceres::abs(point_W[0] * n2_W[0] + point_W[1] * n2_W[1] +
                      point_W[2] * n2_W[2]);

    if (false) {
      T inner_product =
          n1_W[0] * n2_W[0] + n1_W[1] * n2_W[1] + n1_W[2] * n2_W[2];
      residual[0] = (T)weight * ((T)1 - inner_product);
      residual[1] = (T)weight * (d1_W - d2_W);
      residual[2] = (T)0;
      residual[3] = (T)0;
    } else {
      //
      residual[0] = (T)weight * (n1_W[0] * d1_W - n2_W[0] * d2_W);
      residual[1] = (T)weight * (n1_W[1] * d1_W - n2_W[1] * d2_W);
      residual[2] = (T)weight * (n1_W[2] * d1_W - n2_W[2] * d2_W);
      residual[3] = (T)0;
    }

    return true;
  }
};

//! Functor.
struct Cost_function_of_global_plane {
  Eigen::Vector3f _n1_F, _n2_W;
  float _d1_F, _d2_W;
  // Assignment constructor
  Cost_function_of_global_plane(Eigen::Vector3f n1_F, Eigen::Vector3f n2_W,
                                float d1_F, float d2_W)
      : _n1_F(n1_F), _n2_W(n2_W), _d1_F(d1_F), _d2_W(d2_W) {}

  //
  template <typename T>
  bool operator()(const T *const extrinsic_1, T *residual) const {
    //
    T buffer[3], point_W[3], d1_W, d2_W;
    //
    T n1_W[3];
    buffer[0] = T(_n1_F.x());
    buffer[1] = T(_n1_F.y());
    buffer[2] = T(_n1_F.z());
    ceres::AngleAxisRotatePoint(extrinsic_1, buffer, n1_W);
    //
    point_W[0] = -n1_W[0] * (T)_d1_F + extrinsic_1[3];
    point_W[1] = -n1_W[1] * (T)_d1_F + extrinsic_1[4];
    point_W[2] = -n1_W[2] * (T)_d1_F + extrinsic_1[5];
    d1_W =
        -(point_W[0] * n1_W[0] + point_W[1] * n1_W[1] + point_W[2] * n1_W[2]);

    //
    T n2_W[3];
    n2_W[0] = T(_n2_W.x());
    n2_W[1] = T(_n2_W.y());
    n2_W[2] = T(_n2_W.z());
    //
    T inner_product = n1_W[0] * n2_W[0] + n1_W[1] * n2_W[1] + n1_W[2] * n2_W[2];
    //
    d2_W = (T)_d2_W;
    T distance_diff = d1_W - d2_W;

    //
    double weight = 100;
    residual[0] = (T)weight * (n1_W[0] - n2_W[0]);
    residual[1] = (T)weight * (n1_W[1] - n2_W[1]);
    residual[2] = (T)weight * (n1_W[2] - n2_W[2]);
    residual[3] = (T)weight * (distance_diff);

    return true;
  }
};

//! Functor for regular constrain
struct Cost_function_of_regular_constrain {
  float _weight;
  // Assignment constructor
  Cost_function_of_regular_constrain(float weight) : _weight(weight) {}

  //
  template <typename T>
  bool operator()(const T *const extrinsic_1, T *residual) const {
    //
    residual[0] = (T)_weight * ceres::abs(extrinsic_1[0]);
    residual[1] = (T)_weight * ceres::abs(extrinsic_1[1]);
    residual[2] = (T)_weight * ceres::abs(extrinsic_1[2]);
    residual[3] = (T)_weight * ceres::abs(extrinsic_1[3]);
    residual[4] = (T)_weight * ceres::abs(extrinsic_1[4]);
    residual[5] = (T)_weight * ceres::abs(extrinsic_1[5]);

    return true;
  }
};
