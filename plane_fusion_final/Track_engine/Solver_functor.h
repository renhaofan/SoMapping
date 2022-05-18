

// Eigen
#include <float.h>

#include <Eigen/Dense>

//
#define CONDITION_NUMBER_THRESHOLD 300

//
#define CONVERGENCE_STEP 1e-5
#define DIVERGENCE_STEP 1e-1

//
enum IterationState {
  DURING_ITERATION = 0,
  CONVERGENCED = 1,
  FAILED = 2,
};

//! Compute pose matrix from result vector
void compute_pose_mat_from_result(
    const Eigen::Matrix<float, 6, 1> &result_vector,
    Eigen::Matrix4f &output_pose_mat);

//!
/*!


*/
class GaussianNewton_solver {
 public:
  //!
  GaussianNewton_solver() { last_good_pose.setIdentity(); }
  ~GaussianNewton_solver(){};

  //! Store last good incremental pose matrix
  Eigen::Matrix4f last_good_pose;

  //! Overload opertor() as functor usage
  IterationState operator()(const Eigen::Matrix<float, 6, 6> &hessian,
                            const Eigen::Matrix<float, 6, 1> &nabla,
                            Eigen::Matrix<float, 4, 4> &incremental_pose) {
    // Validate hessian is not singular-like matrix
    float condition_number = hessian.lpNorm<Eigen::Infinity>() *
                             hessian.inverse().lpNorm<Eigen::Infinity>();
    if (condition_number > CONDITION_NUMBER_THRESHOLD) {
      incremental_pose = this->last_good_pose;
      printf("condition_number = %f\n", condition_number);
      return IterationState::FAILED;
    }

    // Solve
    Eigen::Matrix<float, 6, 1> result_vector;
    result_vector = hessian.lu().solve(nabla);

    // Compute pose matrix
    Eigen::Matrix4f pose_iterate_inc_mat;
    compute_pose_mat_from_result(result_vector, pose_iterate_inc_mat);

    // Update incremental pose matrix
    incremental_pose = incremental_pose.eval() * pose_iterate_inc_mat;
    this->last_good_pose = incremental_pose;

    //
    if (CONVERGENCE_STEP > result_vector.norm()) {
      return IterationState::CONVERGENCED;
    } else {
      return IterationState::DURING_ITERATION;
    }
  }

  //! Compute pose matrix from result vector
  friend void compute_pose_mat_from_result(
      const Eigen::Matrix<float, 6, 1> &result_vector,
      Eigen::Matrix4f &output_pose_mat);
};

//!
/*!

*/
class LevenbergMarquardt_solver {
 public:
  //!
  LevenbergMarquardt_solver() {
    last_good_pose.setIdentity();
    this->last_average_energy = FLT_MAX;
    this->last_point_pair_number = 0;
    this->damping_coefficient = 1.0f;
    this->identity_mat.setIdentity();
  }
  ~LevenbergMarquardt_solver(){};

  //! Store last good incremental pose matrix
  Eigen::Matrix4f last_good_pose;
  //! Last point average energy
  float last_average_energy;
  //! Last point pair number
  int last_point_pair_number;
  //! Damping coefficient
  float damping_coefficient;
  //! Identity matrix
  Eigen::Matrix<float, 6, 6> identity_mat;

  //!
  const float damping_adjustment = 0.25;

  //! Overload opertor() as functor usage
  IterationState operator()(const Eigen::Matrix<float, 6, 6> &hessian,
                            const Eigen::Matrix<float, 6, 1> &nabla,
                            const float &point_average_distance,
                            const int &points_number,
                            Eigen::Matrix<float, 4, 4> &incremental_pose) {
    // Validate effective optimization step
    if (point_average_distance < this->last_average_energy &&
        points_number >= this->last_point_pair_number * 0.9) {
      this->last_average_energy = point_average_distance;
      this->damping_coefficient *= damping_adjustment;
    } else {
      incremental_pose = this->last_good_pose;
      this->damping_coefficient /= damping_adjustment;
    }

    // Solve
    Eigen::Matrix<float, 6, 6> hessian_lm;
    hessian_lm = hessian + this->identity_mat * this->damping_coefficient;
    //
    Eigen::Matrix<float, 6, 1> result_vector;
    result_vector = hessian_lm.lu().solve(nabla);

    // Compute pose matrix
    Eigen::Matrix4f pose_iterate_inc_mat;
    compute_pose_mat_from_result(result_vector, pose_iterate_inc_mat);

    // Update incremental pose matrix
    incremental_pose = incremental_pose.eval() * pose_iterate_inc_mat;
    this->last_good_pose = incremental_pose;

    //
    if (CONVERGENCE_STEP > result_vector.norm()) {
      return IterationState::CONVERGENCED;
    } else if (result_vector.norm() > DIVERGENCE_STEP) {
      return IterationState::FAILED;
    } else {
      return IterationState::DURING_ITERATION;
    }
  }

  //! Compute pose matrix from result vector
  friend void compute_pose_mat_from_result(
      const Eigen::Matrix<float, 6, 1> &result_vector,
      Eigen::Matrix4f &output_pose_mat);
};
