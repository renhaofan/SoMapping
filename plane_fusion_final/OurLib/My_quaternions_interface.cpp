
//
#include "math.h"
//!
#include "My_quaternions_interface.h"

//! Force compiler make instantation
//! Convert Eigen::Quaternion<T> to My_Type::My_quaternions<T>
template void my_convert<float>(My_Type::My_quaternions<float> &dst_q,
                                const Eigen::Quaternion<float> &src_q);
template void my_convert<double>(My_Type::My_quaternions<double> &dst_q,
                                 const Eigen::Quaternion<double> &src_q);
//! Convert My_Type::My_quaternions<T> to Eigen::Quaternion<T>
template void my_convert<float>(Eigen::Quaternion<float> &dst_q,
                                const My_Type::My_quaternions<float> &src_q);
template void my_convert<double>(Eigen::Quaternion<double> &dst_q,
                                 const My_Type::My_quaternions<double> &src_q);
//! Convert rotation matrix Eigen::Matrix33<T> to My_Type::My_quaternions<T>
template void my_convert(My_Type::My_quaternions<float> &dst_q,
                         const Eigen::Matrix<float, 3, 3> &src_matrix);
template void my_convert(My_Type::My_quaternions<double> &dst_q,
                         const Eigen::Matrix<double, 3, 3> &src_matrix);
//! Convert My_Type::My_quaternions<T> to rotation matrix Eigen::Matrix33<T>
template void my_convert(Eigen::Matrix<float, 3, 3> &dst_matrix,
                         const My_Type::My_quaternions<float> &src_q);
template void my_convert(Eigen::Matrix<double, 3, 3> &dst_matrix,
                         const My_Type::My_quaternions<double> &src_q);

// -------------------------------- My_quaternions To Eigen
// --------------------------------
//
//! Convert Eigen::Quaternion<T> to My_Type::My_quaternions<T>
template <typename T>
void my_convert(My_Type::My_quaternions<T> &dst_q,
                const Eigen::Quaternion<T> &src_q) {
  dst_q.qr = src_q.w();
  dst_q.qx = src_q.x();
  dst_q.qy = src_q.y();
  dst_q.qz = src_q.z();
}

//! Convert My_Type::My_quaternions<T> to Eigen::Quaternion<T>
template <typename T>
void my_convert(Eigen::Quaternion<T> &dst_q,
                const My_Type::My_quaternions<T> &src_q) {
  dst_q.w() = src_q.qr;
  dst_q.x() = src_q.qx;
  dst_q.y() = src_q.qy;
  dst_q.z() = src_q.qz;
}

//! Convert rotation matrix Eigen::Matrix33<T> to My_Type::My_quaternions<T>
template <typename T>
void my_convert(My_Type::My_quaternions<T> &dst_q,
                const Eigen::Matrix<T, 3, 3> &src_matrix) {
  dst_q.qr = (T)sqrtf((double)fabs(1 + src_matrix(0, 0) + src_matrix(1, 1) +
                                   src_matrix(2, 2))) /
             (T)2;
  dst_q.qx = (T)sqrtf((double)fabs(1 + src_matrix(0, 0) - src_matrix(1, 1) -
                                   src_matrix(2, 2))) /
             (T)2;
  dst_q.qy = (T)sqrtf((double)fabs(1 - src_matrix(0, 0) + src_matrix(1, 1) -
                                   src_matrix(2, 2))) /
             (T)2;
  dst_q.qz = (T)sqrtf((double)fabs(1 - src_matrix(0, 0) - src_matrix(1, 1) +
                                   src_matrix(2, 2))) /
             (T)2;
  if (dst_q.qx * dst_q.qr * (src_matrix(2, 1) - src_matrix(1, 2)) < 0)
    dst_q.qx = -dst_q.qx;
  if (dst_q.qy * dst_q.qr * (src_matrix(0, 2) - src_matrix(2, 0)) < 0)
    dst_q.qy = -dst_q.qy;
  if (dst_q.qz * dst_q.qr * (src_matrix(1, 0) - src_matrix(0, 1)) < 0)
    dst_q.qz = -dst_q.qz;
}

//! Convert My_Type::My_quaternions<T> to rotation matrix Eigen::Matrix33<T>
template <typename T>
void my_convert(Eigen::Matrix<T, 3, 3> &dst_matrix,
                const My_Type::My_quaternions<T> &src_q) {
  dst_matrix(0, 0) =
      (T)1 - (T)2 * src_q.qy * src_q.qy - (T)2 * src_q.qz * src_q.qz;
  dst_matrix(1, 1) =
      (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qz * src_q.qz;
  dst_matrix(2, 2) =
      (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qy * src_q.qy;

  dst_matrix(1, 0) = (T)2 * (src_q.qx * src_q.qy + src_q.qr * src_q.qz);
  dst_matrix(0, 1) = (T)2 * (src_q.qx * src_q.qy - src_q.qr * src_q.qz);

  dst_matrix(0, 2) = (T)2 * (src_q.qx * src_q.qz + src_q.qr * src_q.qy);
  dst_matrix(2, 0) = (T)2 * (src_q.qx * src_q.qz - src_q.qr * src_q.qy);

  dst_matrix(2, 1) = (T)2 * (src_q.qz * src_q.qy + src_q.qr * src_q.qx);
  dst_matrix(1, 2) = (T)2 * (src_q.qz * src_q.qy - src_q.qr * src_q.qx);
}
