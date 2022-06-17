

#include "My_pose.h"

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//
My_pose::My_pose() {
  this->mat.setIdentity();
  //
  checkCudaErrors(cudaMalloc((void **)&(this->dev_mat), 16 * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_mat_inv), 16 * sizeof(float)));
  //
  this->synchronize_to_GPU();
}
My_pose::~My_pose() {
  //
  checkCudaErrors(cudaFree(this->dev_mat));
  checkCudaErrors(cudaFree(this->dev_mat_inv));
}

//
void My_pose::load_pose(Trajectory_node &trajectory_node) {
  this->mat.setIdentity();
  // Rotation
  Eigen::Matrix3f rot_matrix;
  my_convert(rot_matrix, trajectory_node.quaternions);
  this->mat.block(0, 0, 3, 3) = rot_matrix;

  // Translation
  this->mat(0, 3) = trajectory_node.tx;
  this->mat(1, 3) = trajectory_node.ty;
  this->mat(2, 3) = trajectory_node.tz;
}

void My_pose::load_pose(const Matrix4f &mat4) {
  this->mat.setIdentity();
  this->mat = mat4;
}

// Synchronize from CPU to GPU
void My_pose::synchronize_to_GPU() {
  Eigen::Matrix4f mat_inv = this->mat.inverse();
  //
  checkCudaErrors(cudaMemcpy(this->dev_mat, this->mat.data(),
                             16 * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(this->dev_mat_inv, mat_inv.data(),
                             16 * sizeof(float), cudaMemcpyHostToDevice));
}

// Synchronize from GPU to CPU
void My_pose::synchronize_from_GPU() {
  checkCudaErrors(cudaMemcpy(this->mat.data(), this->dev_mat,
                             16 * sizeof(float), cudaMemcpyDeviceToHost));
}

//
void My_pose::transformation_Rt_divided(My_pose &pose_transformation) {
  Matrix3f current_rot, transformation_rot;

  // buffer, same to mat.eval()
  transformation_rot = pose_transformation.mat.block(0, 0, 3, 3);
  current_rot = this->mat.block(0, 0, 3, 3);
  //
  this->mat.block(0, 3, 3, 1) =
      this->mat.block(0, 3, 3, 1) - pose_transformation.mat.block(0, 3, 3, 1);
  //
  this->mat.block(0, 0, 3, 3) = current_rot * transformation_rot.inverse();
}
