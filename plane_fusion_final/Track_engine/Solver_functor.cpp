
#include "Track_engine/Solver_functor.h"

#include <float.h>

//! Compute pose matrix from result vector
void compute_pose_mat_from_result(
    const Eigen::Matrix<float, 6, 1> &result_vector,
    Eigen::Matrix4f &output_pose_mat) {
  //
  Eigen::Vector3f rodriguez;
  rodriguez.data()[0] = result_vector.data()[0];  // alpha
  rodriguez.data()[1] = result_vector.data()[1];  // beta
  rodriguez.data()[2] = result_vector.data()[2];  // gamma
  rodriguez.normalize();
  //
  Eigen::Matrix3f rodriguez_cross_mat;
  rodriguez_cross_mat.setZero();
  rodriguez_cross_mat.data()[1] = +rodriguez.data()[2];  // +gamma
  rodriguez_cross_mat.data()[2] = -rodriguez.data()[1];  // -beta
  rodriguez_cross_mat.data()[3] = -rodriguez.data()[2];  // -gamma
  rodriguez_cross_mat.data()[5] = +rodriguez.data()[0];  // +alpha
  rodriguez_cross_mat.data()[6] = +rodriguez.data()[1];  // +beta
  rodriguez_cross_mat.data()[7] = -rodriguez.data()[0];  // -alpha

  //
  Eigen::Matrix3f rot_inc_mat, identity_mat;
  rot_inc_mat.setIdentity();
  identity_mat.setIdentity();
  float theta = result_vector.block(0, 0, 3, 1).norm();

  //
  if (theta > FLT_EPSILON) {
    rot_inc_mat =
        identity_mat + sin(theta) * rodriguez_cross_mat +
        (1 - cos(theta)) * (rodriguez_cross_mat * rodriguez_cross_mat);
  }

  //
  output_pose_mat.setIdentity();
  output_pose_mat.block(0, 0, 3, 3) = rot_inc_mat;
  output_pose_mat.block(0, 3, 3, 1) = result_vector.block(3, 0, 3, 1);
}
