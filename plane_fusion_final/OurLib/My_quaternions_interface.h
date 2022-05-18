//!
/*!


*/

#pragma once

//!
#include "My_quaternions.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

//! Convert Eigen::Quaternion<T> to My_Type::My_quaternions<T>
template <typename T>
void my_convert(My_Type::My_quaternions<T> &dst_q,
                const Eigen::Quaternion<T> &src_q);

//! Convert My_Type::My_quaternions<T> to Eigen::Quaternion<T>
template <typename T>
void my_convert(Eigen::Quaternion<T> &dst_q,
                const My_Type::My_quaternions<T> &src_q);

//! Convert rotation matrix Eigen::Matrix33<T> to My_Type::My_quaternions<T>
template <typename T>
void my_convert(My_Type::My_quaternions<T> &dst_q,
                const Eigen::Matrix<T, 3, 3> &src_matrix);

//! Convert My_Type::My_quaternions<T> to rotation matrix Eigen::Matrix33<T>
template <typename T>
void my_convert(Eigen::Matrix<T, 3, 3> &dst_matrix,
                const My_Type::My_quaternions<T> &src_q);
