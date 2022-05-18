
#pragma once

#include "My_quaternions.h"
#include "My_vector.h"

// CUDA or Cpp code
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _IS_CUDA_CODE_ __device__
#else
#define _IS_CUDA_CODE_
#endif

namespace My_Type {
// matrix 2x2
template <typename T> struct _Matrix22 {
  union {
    struct {
      T m00, m10; // Index: |0, 1|  Matrix: |m00(0), m01(2)|
      T m01, m11; //		  |2, 3|		  |m10(1), m11(3)|
    };
    T data[4];
  };
};

// matrix 3x3
template <typename T> struct _Matrix33 {
  union {
    struct {
      T m00, m10, m20; // Index: |0, 1, 2|  Matrix: |m00(0), m01(3), m02(6)|
      T m01, m11, m21; //		  |3, 4, 5|			 |m10(1),
                       //m11(4), m12(7)|
      T m02, m12, m22; //		  |6, 7, 8|			 |m20(2),
                       //m21(5), m22(8)|
    };
    T data[9];
  };
};

// matrix 4x4
template <typename T> struct _Matrix44 {
  union {
    struct {
      //
      //! cols frist order matrix
      T m00, m10, m20, m30; // Index: |0,	1,	2,  3 |  Matrix: |m00(0), m01(4),
                            // m02(8),  m03(12)|
      T m01, m11, m21, m31; //		  |4,	5,	6,  7 |			 |m10(1), m11(5),
                            //m12(9),  m13(13)|
      T m02, m12, m22, m32; //		  |8,	9,	10, 11|			 |m20(2), m21(6),
                            //m22(10), m23(14)|
      T m03, m13, m23, m33; //		  |12,	13, 14, 15|			 |m30(3), m31(7),
                            //m32(11), m33(15)|
    };
    T data[16];
  };
};

// matrix MxN
template <typename T> struct _MatrixMN {
  int cols, rows;
  T *data;
};

//
template <typename T> class Matrix22 : public _Matrix22<T> {
public:
  _IS_CUDA_CODE_ Matrix22() {}
  _IS_CUDA_CODE_ ~Matrix22() {}

  //! Set matrix to a identity matrix.
  _IS_CUDA_CODE_ inline void set_identity() {
    this->m00 = 1;
    this->m01 = 0;
    this->m10 = 0;
    this->m11 = 1;
  }
  //! Set matrix to zero
  _IS_CUDA_CODE_ inline void set_zero() {
    for (int i = 0; i < 4; i++)
      this->data[i] = 0;
  }

  //! Over load operator[]
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //! Over load operator()
  _IS_CUDA_CODE_ T &operator()(int row, int col) {
    return this->data[row + col * 2];
  }

  // Over load '='
  _IS_CUDA_CODE_ Matrix22<T> &operator=(const Matrix22<T> &mat) {
    for (int i = 0; i < 4; i++)
      this->data[i] = mat.data[i];
    return (*this);
  }

  //
  _IS_CUDA_CODE_ inline Matrix22<T> &operator+=(const T &r) {
    for (int i = 0; i < 4; i++)
      this->data[i] += r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix22<T> &operator-=(const T &r) {
    for (int i = 0; i < 4; i++)
      this->data[i] -= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix22<T> &operator*=(const T &r) {
    for (int i = 0; i < 4; i++)
      this->data[i] *= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix22<T> &operator/=(const T &r) {
    for (int i = 0; i < 4; i++)
      this->data[i] /= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix22<T> &operator+=(const Matrix22<T> &mat) {
    for (int i = 0; i < 4; i++)
      this->data[i] += mat.data[i];
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix22<T> &operator-=(const Matrix22<T> &mat) {
    for (int i = 0; i < 4; i++)
      this->data[i] -= mat.data[i];
    return (*this);
  }

  //! Matrix22 multiply Matrix22
  _IS_CUDA_CODE_ inline friend Matrix22<T> operator*(Matrix22<T> &mat_1,
                                                     Matrix22<T> &mat_2) {
    Matrix22<T> mat_r;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        T temp_sum = 0;
        for (int k = 0; k < 2; k++) {
          temp_sum += mat_1[i + k * 2] * mat_2[k + j * 2];
        }
        mat_r[i + j * 2] = temp_sum;
      }
    }

    return mat_r;
  }

  //! Matrix22 multiply Vector2
  _IS_CUDA_CODE_ inline friend Vector2<T> operator*(Matrix22<T> &mat,
                                                    Vector2<T> &vec) {
    Vector2<T> vec_r;
    for (int i = 0; i < 2; i++) {
      T temp_sum = 0;
      for (int k = 0; k < 2; k++) {
        temp_sum += mat[i + k * 2] * vec[k];
      }
      vec_r[i] = temp_sum;
    }
    return vec_r;
  }

  //!
  friend std::ostream &operator<<(std::ostream &out_stream,
                                  const Matrix22<T> &mat) {
    out_stream << mat(0, 0) << ", " << mat(0, 1) << "\r\n"
               << mat(1, 0) << ", " << mat(1, 1);
    return out_stream;
  }
};

//
template <typename T> class Matrix33 : public _Matrix33<T> {
public:
  _IS_CUDA_CODE_ Matrix33() {}
  _IS_CUDA_CODE_ ~Matrix33() {}

  //! Set matrix to a identity matrix.
  _IS_CUDA_CODE_ inline void set_identity() {
    this->m00 = 1;
    this->m01 = 0;
    this->m02 = 0;
    this->m10 = 0;
    this->m11 = 1;
    this->m12 = 0;
    this->m20 = 0;
    this->m21 = 0;
    this->m22 = 1;
  }

  //! Set matrix to zero
  _IS_CUDA_CODE_ inline void set_zero() {
    for (int i = 0; i < 9; i++)
      this->data[i] = 0;
  }

  //! Over load operator[]
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //! Over load operator()
  _IS_CUDA_CODE_ T &operator()(int row, int col) {
    return this->data[row + col * 3];
  }

  //! Load rotation from quaternions (My_Type::My_quaternions<T>)
  _IS_CUDA_CODE_ Matrix33<T> &load_from_quaternions(My_quaternions<T> &src_q) {
    (*this)(0, 0) =
        (T)1 - (T)2 * src_q.qy * src_q.qy - (T)2 * src_q.qz * src_q.qz;
    (*this)(1, 1) =
        (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qz * src_q.qz;
    (*this)(2, 2) =
        (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qy * src_q.qy;

    (*this)(1, 0) = (T)2 * (src_q.qx * src_q.qy + src_q.qr * src_q.qz);
    (*this)(0, 1) = (T)2 * (src_q.qx * src_q.qy - src_q.qr * src_q.qz);

    (*this)(0, 2) = (T)2 * (src_q.qx * src_q.qz + src_q.qr * src_q.qy);
    (*this)(2, 0) = (T)2 * (src_q.qx * src_q.qz - src_q.qr * src_q.qy);

    (*this)(2, 1) = (T)2 * (src_q.qz * src_q.qy + src_q.qr * src_q.qx);
    (*this)(1, 2) = (T)2 * (src_q.qz * src_q.qy - src_q.qr * src_q.qx);

    return (*this);
  }

  //! Convert to quaternions (My_Type::My_quaternions<T>)
  _IS_CUDA_CODE_ My_quaternions<T> convert_to_quaternions() {
    My_quaternions<T> dst_q;

    dst_q.qr = (T)sqrt((double)abs(1 + (*this)(0, 0) + (*this)(1, 1) +
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qx = (T)sqrt((double)abs(1 + (*this)(0, 0) - (*this)(1, 1) -
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qy = (T)sqrt((double)abs(1 - (*this)(0, 0) + (*this)(1, 1) -
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qz = (T)sqrt((double)abs(1 - (*this)(0, 0) - (*this)(1, 1) +
                                   (*this)(2, 2))) /
               (T)2;
    if (dst_q.qx * dst_q.qr * ((*this)(2, 1) - (*this)(1, 2)) < 0)
      dst_q.qx = -dst_q.qx;
    if (dst_q.qy * dst_q.qr * ((*this)(0, 2) - (*this)(2, 0)) < 0)
      dst_q.qy = -dst_q.qy;
    if (dst_q.qz * dst_q.qr * ((*this)(1, 0) - (*this)(0, 1)) < 0)
      dst_q.qz = -dst_q.qz;

    return dst_q;
  }

  _IS_CUDA_CODE_ Matrix33<T> &operator=(const Matrix33<T> &mat) {
    for (int i = 0; i < 9; i++)
      this->data[i] = mat.data[i];
    return (*this);
  }

  _IS_CUDA_CODE_ inline Matrix33<T> &operator+=(const T &r) {
    for (int i = 0; i < 9; i++)
      this->data[i] += r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix33<T> &operator-=(const T &r) {
    for (int i = 0; i < 9; i++)
      this->data[i] -= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix33<T> &operator*=(const T &r) {
    for (int i = 0; i < 9; i++)
      this->data[i] *= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix33<T> &operator/=(const T &r) {
    for (int i = 0; i < 9; i++)
      this->data[i] /= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix33<T> &operator+=(const Matrix33<T> &mat) {
    for (int i = 0; i < 9; i++)
      this->data[i] += mat.data[i];
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix33<T> &operator-=(const Matrix33<T> &mat) {
    for (int i = 0; i < 9; i++)
      this->data[i] -= mat.data[i];
    return (*this);
  }

  //! Matrix33 multiply Matrix33
  _IS_CUDA_CODE_ inline friend Matrix33<T> operator*(Matrix33<T> &mat_1,
                                                     Matrix33<T> &mat_2) {
    Matrix33<T> mat_r;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        T temp_sum = 0;
        for (int k = 0; k < 3; k++) {
          temp_sum += mat_1[i + k * 3] * mat_2[k + j * 3];
        }
        mat_r[i + j * 3] = temp_sum;
      }
    }

    return mat_r;
  }

  //! Matrix33 multiply Vector3
  _IS_CUDA_CODE_ inline friend Vector3<T> operator*(Matrix33<T> &mat,
                                                    Vector3<T> &vec) {
    Vector3<T> vec_r;
    for (int i = 0; i < 3; i++) {
      T temp_sum = 0;
      for (int k = 0; k < 3; k++) {
        temp_sum += mat[i + k * 3] * vec[k];
      }
      vec_r[i] = temp_sum;
    }
    return vec_r;
  }

  //!
  friend std::ostream &operator<<(std::ostream &out_stream, Matrix33<T> &mat) {
    out_stream << mat(0, 0) << ", " << mat(0, 1) << ", " << mat(0, 2) << "\r\n"
               << mat(1, 0) << ", " << mat(1, 1) << ", " << mat(1, 2) << "\r\n"
               << mat(2, 0) << ", " << mat(2, 1) << ", " << mat(2, 2);
    return out_stream;
  }
};

//
template <typename T> class Matrix44 : public _Matrix44<T> {
public:
  _IS_CUDA_CODE_ Matrix44() {}
  _IS_CUDA_CODE_ ~Matrix44() {}

  //! Set matrix to a identity matrix.
  _IS_CUDA_CODE_ inline void set_identity() {
    this->m00 = 1;
    this->m01 = 0;
    this->m02 = 0;
    this->m03 = 0;
    this->m10 = 0;
    this->m11 = 1;
    this->m12 = 0;
    this->m13 = 0;
    this->m20 = 0;
    this->m21 = 0;
    this->m22 = 1;
    this->m23 = 0;
    this->m30 = 0;
    this->m31 = 0;
    this->m32 = 0;
    this->m33 = 1;
  }
  //! Set matrix to zero
  _IS_CUDA_CODE_ inline void set_zero() {
    for (int i = 0; i < 16; i++)
      this->data[i] = 0;
  }

  //! Over load operator[]
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //!
  _IS_CUDA_CODE_ T &operator()(int row, int col) {
    return this->data[row + col * 4];
  }

  //! Load rotation from quaternions (My_Type::My_quaternions<T>)
  _IS_CUDA_CODE_ Matrix44<T> &load_from_quaternions(My_quaternions<T> &src_q) {
    (*this)(0, 0) =
        (T)1 - (T)2 * src_q.qy * src_q.qy - (T)2 * src_q.qz * src_q.qz;
    (*this)(1, 1) =
        (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qz * src_q.qz;
    (*this)(2, 2) =
        (T)1 - (T)2 * src_q.qx * src_q.qx - (T)2 * src_q.qy * src_q.qy;

    (*this)(1, 0) = (T)2 * (src_q.qx * src_q.qy + src_q.qr * src_q.qz);
    (*this)(0, 1) = (T)2 * (src_q.qx * src_q.qy - src_q.qr * src_q.qz);

    (*this)(0, 2) = (T)2 * (src_q.qx * src_q.qz + src_q.qr * src_q.qy);
    (*this)(2, 0) = (T)2 * (src_q.qx * src_q.qz - src_q.qr * src_q.qy);

    (*this)(2, 1) = (T)2 * (src_q.qz * src_q.qy + src_q.qr * src_q.qx);
    (*this)(1, 2) = (T)2 * (src_q.qz * src_q.qy - src_q.qr * src_q.qx);

    return (*this);
  }

  //! Convert to quaternions (My_Type::My_quaternions<T>)
  _IS_CUDA_CODE_ My_quaternions<T> convert_to_quaternions() {
    My_quaternions<T> dst_q;

    dst_q.qr = (T)sqrt((double)abs(1 + (*this)(0, 0) + (*this)(1, 1) +
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qx = (T)sqrt((double)abs(1 + (*this)(0, 0) - (*this)(1, 1) -
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qy = (T)sqrt((double)abs(1 - (*this)(0, 0) + (*this)(1, 1) -
                                   (*this)(2, 2))) /
               (T)2;
    dst_q.qz = (T)sqrt((double)abs(1 - (*this)(0, 0) - (*this)(1, 1) +
                                   (*this)(2, 2))) /
               (T)2;
    if (dst_q.qx * dst_q.qr * ((*this)(2, 1) - (*this)(1, 2)) < 0)
      dst_q.qx = -dst_q.qx;
    if (dst_q.qy * dst_q.qr * ((*this)(0, 2) - (*this)(2, 0)) < 0)
      dst_q.qy = -dst_q.qy;
    if (dst_q.qz * dst_q.qr * ((*this)(1, 0) - (*this)(0, 1)) < 0)
      dst_q.qz = -dst_q.qz;

    return dst_q;
  }

  _IS_CUDA_CODE_ Matrix44<T> &operator=(const Matrix44<T> &mat) {
    for (int i = 0; i < 16; i++)
      this->data[i] = mat.data[i];
    return (*this);
  }

  _IS_CUDA_CODE_ inline Matrix44<T> &operator+=(const T &r) {
    for (int i = 0; i < 16; i++)
      this->data[i] += r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix44<T> &operator-=(const T &r) {
    for (int i = 0; i < 16; i++)
      this->data[i] -= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix44<T> &operator*=(const T &r) {
    for (int i = 0; i < 16; i++)
      this->data[i] *= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix44<T> &operator/=(const T &r) {
    for (int i = 0; i < 16; i++)
      this->data[i] /= r;
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix44<T> &operator+=(const Matrix44<T> &mat) {
    for (int i = 0; i < 16; i++)
      this->data[i] += mat.data[i];
    return (*this);
  }
  _IS_CUDA_CODE_ inline Matrix44<T> &operator-=(const Matrix44<T> &mat) {
    for (int i = 0; i < 16; i++)
      this->data[i] -= mat.data[i];
    return (*this);
  }

  //! Matrix44 multiply Matrix44
  _IS_CUDA_CODE_ inline friend Matrix44<T> operator*(Matrix44<T> &mat_1,
                                                     Matrix44<T> &mat_2) {
    Matrix44 mat_r;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        T temp_sum = 0;
        for (int k = 0; k < 4; k++) {
          temp_sum += mat_1[i + k * 4] * mat_2[k + j * 4];
        }
        mat_r[i + j * 4] = temp_sum;
      }
    }
    return mat_r;
  }

  //! Matrix44 multiply Vector4
  _IS_CUDA_CODE_ inline friend Vector4<T> operator*(Matrix44<T> &mat,
                                                    Vector4<T> &vec) {
    Vector4<T> vec_r;
    for (int i = 0; i < 4; i++) {
      T temp_sum = 0;
      for (int k = 0; k < 4; k++) {
        temp_sum += mat[i + k * 4] * vec[k];
      }
      vec_r[i] = temp_sum;
    }
    return vec_r;
  }

  //!
  friend std::ostream &operator<<(std::ostream &out_stream, Matrix44<T> &mat) {
    out_stream << mat(0, 0) << ", " << mat(0, 1) << ", " << mat(0, 2) << ", "
               << mat(0, 3) << "\r\n"
               << mat(1, 0) << ", " << mat(1, 1) << ", " << mat(1, 2) << ", "
               << mat(1, 3) << "\r\n"
               << mat(2, 0) << ", " << mat(2, 1) << ", " << mat(2, 2) << ", "
               << mat(2, 3) << "\r\n"
               << mat(3, 0) << ", " << mat(3, 1) << ", " << mat(3, 2) << ", "
               << mat(3, 3);
    return out_stream;
  }
};

// Reference: Eigen 3.2
#define MAKE_TYPES(Type, TypeSuffix)                                           \
  typedef Matrix22<Type> Matrix22##TypeSuffix;                                 \
  typedef Matrix33<Type> Matrix33##TypeSuffix;                                 \
  typedef Matrix44<Type> Matrix44##TypeSuffix;

MAKE_TYPES(float, f);
MAKE_TYPES(double, d);
MAKE_TYPES(char, c);
MAKE_TYPES(unsigned char, uc);
MAKE_TYPES(short, s);
MAKE_TYPES(unsigned short, us);
MAKE_TYPES(int, i);
MAKE_TYPES(unsigned int, ui);

#undef MAKE_TYPES

} // namespace My_Type
