//!
/*


*/

#pragma once

//!
#include <iostream>

//      （CUDA   NVCC   C++   ）
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _IS_CUDA_CODE_ __device__
#else
#define _IS_CUDA_CODE_
#endif

namespace My_Type {

//! My quarernions template class. Only defined float type and double type.
/*!


*/
template <typename T>
class My_quaternions {
 public:
  //! Real, image X, image Y, image Z
  T qr, qx, qy, qz;

  //! Constructor
  _IS_CUDA_CODE_ My_quaternions() {
    // Set rotation identity
    this->qr = (T)1;
    this->qx = (T)0;
    this->qy = (T)0;
    this->qz = (T)0;
  };

  //! Destructor
  _IS_CUDA_CODE_ ~My_quaternions() {}

  //! Copy constructor
  _IS_CUDA_CODE_ My_quaternions(const My_quaternions &quaternions_vec) {
    this->qr = quaternions_vec.qr;
    this->qx = quaternions_vec.qx;
    this->qy = quaternions_vec.qy;
    this->qz = quaternions_vec.qz;
  }

  //! Assignment constructor
  _IS_CUDA_CODE_ My_quaternions(T _qr, T _qx, T _qy, T _qz)
      : qr(_qr), qx(_qx), qy(_qy), qz(_qz) {}

  //! Set quaternions identity
  _IS_CUDA_CODE_ inline void set_identity() {
    // Set rotation identity
    this->qr = (T)1;
    this->qx = (T)0;
    this->qy = (T)0;
    this->qz = (T)0;
  }

  //! Norm
  _IS_CUDA_CODE_ inline T norm() {
    T norm_value = this->qr * this->qr + this->qx * this->qx +
                   this->qy * this->qy + this->qz * this->qz;
    norm_value = (T)sqrt((double)abs(norm_value));

    return norm_value;
  }

  //! Normalize
  _IS_CUDA_CODE_ inline void normalize() {
    // Compute norm
    T norm_value = this->norm();
    // Normalize
    this->qr /= norm_value;
    this->qx /= norm_value;
    this->qy /= norm_value;
    this->qz /= norm_value;

    return;
  }

  //! Quaternions conjugate
  _IS_CUDA_CODE_ inline My_quaternions conjugate() {
    My_quaternions<T> result(this->qr, -this->qx, -this->qy, -this->qz);

    return result;
  }

  //! Quaternions inverse
  _IS_CUDA_CODE_ inline My_quaternions inverse() {
    My_quaternions<T> result(this->qr, -this->qx, -this->qy, -this->qz);

    result *= (T)1.0 / (T)result.norm();
    return result;
  }

  //! Quaternions "Quaternions += Scalar"
  _IS_CUDA_CODE_ inline My_quaternions &operator+=(const T &scalar_1) {
    this->qr += scalar_1;
    this->qx += scalar_1;
    this->qy += scalar_1;
    this->qz += scalar_1;

    return (*this);
  }
  //! Quaternions "Quaternions -= Scalar"
  _IS_CUDA_CODE_ inline My_quaternions &operator-=(const T &scalar_1) {
    this->qr -= scalar_1;
    this->qx -= scalar_1;
    this->qy -= scalar_1;
    this->qz -= scalar_1;

    return (*this);
  }
  //! Quaternions "Quaternions *= Scalar"
  _IS_CUDA_CODE_ inline My_quaternions &operator*=(const T &scalar_1) {
    //
    this->qr *= scalar_1;
    this->qx *= scalar_1;
    this->qy *= scalar_1;
    this->qz *= scalar_1;

    return (*this);
  }

  //! Quaternions "+="
  _IS_CUDA_CODE_ inline My_quaternions &operator+=(
      const My_quaternions &quaternions_1) {
    this->qr += quaternions_1.qr;
    this->qx += quaternions_1.qx;
    this->qy += quaternions_1.qy;
    this->qz += quaternions_1.qz;

    return (*this);
  }
  //! Quaternions "-="
  _IS_CUDA_CODE_ inline My_quaternions &operator-=(
      const My_quaternions &quaternions_1) {
    this->qr -= quaternions_1.qr;
    this->qx -= quaternions_1.qx;
    this->qy -= quaternions_1.qy;
    this->qz -= quaternions_1.qz;

    return (*this);
  };
  //! Quaternions "*="
  _IS_CUDA_CODE_ inline My_quaternions &operator*=(
      const My_quaternions &quaternions_1) {
    My_quaternions<T> buffer(*this);
    //
    this->qr = buffer.qr * quaternions_1.qr - buffer.qx * quaternions_1.qx -
               buffer.qy * quaternions_1.qy - buffer.qz * quaternions_1.qz;
    this->qx = buffer.qr * quaternions_1.qx + buffer.qx * quaternions_1.qr +
               buffer.qy * quaternions_1.qz - buffer.qz * quaternions_1.qy;
    this->qy = buffer.qr * quaternions_1.qy - buffer.qx * quaternions_1.qz +
               buffer.qy * quaternions_1.qr + buffer.qz * quaternions_1.qx;
    this->qz = buffer.qr * quaternions_1.qz + buffer.qx * quaternions_1.qy -
               buffer.qy * quaternions_1.qx + buffer.qz * quaternions_1.qr;

    return (*this);
  }

  //! Quaternions "+"
  _IS_CUDA_CODE_ inline friend My_quaternions operator+(
      const My_quaternions &quaternions_1,
      const My_quaternions &quaternions_2) {
    My_quaternions<T> result(quaternions_1);
    result += quaternions_2;

    return (result);
  }
  //! Quaternions "-"
  _IS_CUDA_CODE_ inline friend My_quaternions operator-(
      const My_quaternions &quaternions_1,
      const My_quaternions &quaternions_2) {
    My_quaternions<T> result(quaternions_1);
    result -= quaternions_2;

    return (result);
  }
  //! Quaternions "*"
  _IS_CUDA_CODE_ inline friend My_quaternions operator*(
      const My_quaternions &quaternions_1,
      const My_quaternions &quaternions_2) {
    My_quaternions<T> result(quaternions_1);
    result *= quaternions_2;

    return (result);
  }

  //! Quaternions "*" Scalar
  _IS_CUDA_CODE_ inline friend My_quaternions operator*(
      const float &scalar, const My_quaternions &quaternions_1) {
    My_quaternions<T> result(quaternions_1);
    result *= scalar;

    return (result);
  }
  //! Scalar "*" Quaternions
  _IS_CUDA_CODE_ inline friend My_quaternions operator*(
      const My_quaternions &quaternions_1, const float &scalar) {
    My_quaternions<T> result(quaternions_1);
    result *= scalar;

    return (result);
  }

  //! Print quaternions information
  _IS_CUDA_CODE_ inline void print() {
    printf("%f, %f, %f, %f\r\n", (float)this->qr, (float)this->qx,
           (float)this->qy, (float)this->qz);
    return;
  }

  //!
  friend std::ostream &operator<<(std::ostream &out_stream,
                                  const My_quaternions &quaternions_1) {
    out_stream << quaternions_1.qr << ", " << quaternions_1.qx << ", "
               << quaternions_1.qy << ", " << quaternions_1.qz;
    return out_stream;
  }
};

// pre-define
typedef My_quaternions<float> My_quaternionsf;
typedef My_quaternions<double> My_quaternionsd;

}  // namespace My_Type
