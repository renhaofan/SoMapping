

#pragma once

//
#include <math.h>

#include <iostream>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
//#if defined(__global__)
#define _IS_CUDA_CODE_ __device__
#else
#define _IS_CUDA_CODE_
#endif

//
namespace My_Type {

// Vector-2 structure
template <typename T>
struct Vector2_ {
  union {
    struct {
      T x, y;
    };
    struct {
      T u, v;
    };
    struct {
      T width, height;
    };
    T data[2];
  };
};

// Vector-3 structure
template <typename T>
struct Vector3_ {
  union {
    struct {
      T x, y, z;
    };
    struct {
      T nx, ny, nz;
    };
    struct {
      T r, g, b;
    };
    T data[3];
  };
};

// Vector-4 structure
template <typename T>
struct Vector4_ {
  union {
    struct {
      T x, y, z, w;
    };
    struct {
      T r, g, b, a;
    };
    T data[4];
  };
};

// Vector-X structure
template <typename T, int Length>
struct VectorX_ {
  T data[Length];
};

//! Vector-2 template class
template <typename T>
class Vector2 : public Vector2_<T> {
 public:
  //
  _IS_CUDA_CODE_ Vector2(){};
  // explicit
  _IS_CUDA_CODE_ explicit Vector2(const T &value) {
    this->x = value;
    this->y = value;
  }
  _IS_CUDA_CODE_ explicit Vector2(const T *data) {
    this->x = data[0];
    this->y = data[1];
  }
  _IS_CUDA_CODE_ Vector2(const T v0, const T v1) {
    this->x = v0;
    this->y = v1;
  }
  _IS_CUDA_CODE_ Vector2(const Vector2_<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
  }

  //
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //      =
  _IS_CUDA_CODE_ Vector2<T> &operator=(const T value) {
    this->x = value;
    this->y = value;
    return (*this);
  }
  _IS_CUDA_CODE_ Vector2<T> &operator=(const Vector2<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
    return (*this);
  }

  // norm
  _IS_CUDA_CODE_ inline T norm() {
    T norm_len = (T)sqrt(this->x * this->x + this->y * this->y);
    return norm_len;
  }
  // inner product
  _IS_CUDA_CODE_ inline T dot(const Vector2<T> &vec) {
    T inner_product = (T)(this->x * vec.x + this->y * vec.y);
    return inner_product;
  }
  // normalize
  _IS_CUDA_CODE_ inline void normlize() {
    T norm_len = this->norm();
    this->x /= norm_len;
    this->y /= norm_len;
  }

  //      （    ，     ！）
  //    += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator+=(Vector2<T> &vec,
                                                      const T value) {
    vec.x += value;
    vec.y += value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator-=(Vector2<T> &vec,
                                                      const T value) {
    vec.x -= value;
    vec.y -= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator*=(Vector2<T> &vec,
                                                      const T value) {
    vec.x *= value;
    vec.y *= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator/=(Vector2<T> &vec,
                                                      const T value) {
    vec.x /= value;
    vec.y /= value;
    return vec;
  }
  //      （        ） += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator+=(Vector2<T> &vec_l,
                                                      const Vector2<T> &vec_r) {
    vec_l.x += vec_r.x;
    vec_l.y += vec_r.y;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator-=(Vector2<T> &vec_l,
                                                      const Vector2<T> &vec_r) {
    vec_l.x -= vec_r.x;
    vec_l.y -= vec_r.y;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator*=(Vector2<T> &vec_l,
                                                      const Vector2<T> &vec_r) {
    vec_l.x *= vec_r.x;
    vec_l.y *= vec_r.y;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> &operator/=(Vector2<T> &vec_l,
                                                      const Vector2<T> &vec_r) {
    vec_l.x /= vec_r.x;
    vec_l.y /= vec_r.y;
    return vec_l;
  }
  //    + - * /
  _IS_CUDA_CODE_ inline friend Vector2<T> operator+(const Vector2<T> &vec_l,
                                                    T value) {
    Vector2<T> vec_return(vec_l);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator+(T value,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_r);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator-(const Vector2<T> &vec_l,
                                                    T value) {
    Vector2<T> vec_return(vec_l);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator-(T value,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_r);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator*(const Vector2<T> &vec_l,
                                                    T value) {
    Vector2<T> vec_return(vec_l);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator*(T value,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_r);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator/(const Vector2<T> &vec_l,
                                                    T value) {
    Vector2<T> vec_return(vec_l);
    return (vec_return /= value);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator/(T value,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_r);
    return (vec_return /= value);
  }
  //      （        ） + - * /
  _IS_CUDA_CODE_ inline friend Vector2<T> operator+(const Vector2<T> &vec_l,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_l);
    return (vec_return += vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator-(const Vector2<T> &vec_l,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_l);
    return (vec_return -= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator*(const Vector2<T> &vec_l,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_l);
    return (vec_return *= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector2<T> operator/(const Vector2<T> &vec_l,
                                                    const Vector2<T> &vec_r) {
    Vector2<T> vec_return(vec_l);
    return (vec_return /= vec_r);
  }

  //
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector2<T> &vec_l,
                                               const T &value) {
    return (vec_l.x == value) && (vec_l.y == value);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector2<T> &vec_l,
                                               const T &value) {
    return (vec_l.x != value) || (vec_l.y != value);
  }
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector2<T> &vec_l,
                                               const Vector2<T> &vec_r) {
    return (vec_l.x == vec_r.x) && (vec_l.y == vec_r.y);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector2<T> &vec_l,
                                               const Vector2<T> &vec_r) {
    return (vec_l.x != vec_r.x) || (vec_l.y != vec_r.y);
  }

  //
  _IS_CUDA_CODE_ inline friend std::ostream &operator<<(std::ostream &os,
                                                        const Vector2<T> &vec) {
    os << vec.x << ", " << vec.y;
    return os;
  }
};
//! Vector-3 template class
template <typename T>
class Vector3 : public Vector3_<T> {
 public:
  //
  _IS_CUDA_CODE_ Vector3(){};
  // explicit
  _IS_CUDA_CODE_ explicit Vector3(const T &value) {
    this->x = value;
    this->y = value;
    this->z = value;
  }
  _IS_CUDA_CODE_ explicit Vector3(const T *data) {
    this->x = data[0];
    this->y = data[1];
    this->z = data[2];
  }
  _IS_CUDA_CODE_ Vector3(const T v0, const T v1, const T v2) {
    this->x = v0;
    this->y = v1;
    this->z = v2;
  }
  _IS_CUDA_CODE_ Vector3(const Vector3_<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
  }

  //
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  _IS_CUDA_CODE_ Vector3<T> &operator=(const T value) {
    this->x = value;
    this->y = value;
    this->z = value;
    return (*this);
  }
  _IS_CUDA_CODE_ Vector3<T> &operator=(const Vector3<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    return (*this);
  }

  // norm
  _IS_CUDA_CODE_ inline T norm() {
    T norm_len =
        (T)sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
    return norm_len;
  }
  // inner product
  _IS_CUDA_CODE_ inline T dot(const Vector3<T> &vec) {
    T inner_product = (T)(this->x * vec.x + this->y * vec.y + this->z * vec.z);
    return inner_product;
  }
  // Cross product
  _IS_CUDA_CODE_ inline Vector3<T> cross(const Vector3<T> &vec) {
    Vector3<T> cross_vec;
    cross_vec.x = this->y * vec.z - this->z * vec.y;
    cross_vec.y = this->z * vec.x - this->x * vec.z;
    cross_vec.z = this->x * vec.y - this->y * vec.x;

    return cross_vec;
  }
  // normalize
  _IS_CUDA_CODE_ inline void normlize() {
    T norm_len = this->norm();
    this->x /= norm_len;
    this->y /= norm_len;
    this->z /= norm_len;
  }

  //      （    ，     ！）
  //    += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator+=(Vector3<T> &vec,
                                                      T value) {
    vec.x += value;
    vec.y += value;
    vec.z += value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator-=(Vector3<T> &vec,
                                                      T value) {
    vec.x -= value;
    vec.y -= value;
    vec.z -= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator*=(Vector3<T> &vec,
                                                      T value) {
    vec.x *= value;
    vec.y *= value;
    vec.z *= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator/=(Vector3<T> &vec,
                                                      T value) {
    vec.x /= value;
    vec.y /= value;
    vec.z /= value;
    return vec;
  }
  //      （        ） += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator+=(Vector3<T> &vec_l,
                                                      const Vector3<T> &vec_r) {
    vec_l.x += vec_r.x;
    vec_l.y += vec_r.y;
    vec_l.z += vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator-=(Vector3<T> &vec_l,
                                                      const Vector3<T> &vec_r) {
    vec_l.x -= vec_r.x;
    vec_l.y -= vec_r.y;
    vec_l.z -= vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator*=(Vector3<T> &vec_l,
                                                      const Vector3<T> &vec_r) {
    vec_l.x *= vec_r.x;
    vec_l.y *= vec_r.y;
    vec_l.z *= vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> &operator/=(Vector3<T> &vec_l,
                                                      const Vector3<T> &vec_r) {
    vec_l.x /= vec_r.x;
    vec_l.y /= vec_r.y;
    vec_l.z /= vec_r.z;
    return vec_l;
  }
  // Volatile variables ----------------------- TODO : chech this for volatile
  // variables!
  _IS_CUDA_CODE_ inline friend volatile Vector3<T> &operator+=(
      volatile Vector3<T> &vec_l, volatile Vector3<T> &vec_r) {
    vec_l.x += vec_r.x;
    vec_l.y += vec_r.y;
    vec_l.z += vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend volatile Vector3<T> &operator-=(
      volatile Vector3<T> &vec_l, volatile Vector3<T> &vec_r) {
    vec_l.x -= vec_r.x;
    vec_l.y -= vec_r.y;
    vec_l.z -= vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend volatile Vector3<T> &operator*=(
      volatile Vector3<T> &vec_l, volatile Vector3<T> &vec_r) {
    vec_l.x *= vec_r.x;
    vec_l.y *= vec_r.y;
    vec_l.z *= vec_r.z;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend volatile Vector3<T> &operator/=(
      volatile Vector3<T> &vec_l, volatile Vector3<T> &vec_r) {
    vec_l.x /= vec_r.x;
    vec_l.y /= vec_r.y;
    vec_l.z /= vec_r.z;
    return vec_l;
  }

  //    + - * /
  _IS_CUDA_CODE_ inline friend Vector3<T> operator+(const Vector3<T> &vec_l,
                                                    T value) {
    Vector3<T> vec_return(vec_l);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator+(T value,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_r);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator-(const Vector3<T> &vec_l,
                                                    T value) {
    Vector3<T> vec_return(vec_l);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator-(T value,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_r);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator*(const Vector3<T> &vec_l,
                                                    T value) {
    Vector3<T> vec_return(vec_l);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator*(T value,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_r);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator/(const Vector3<T> &vec_l,
                                                    T value) {
    Vector3<T> vec_return(vec_l);
    return (vec_return /= value);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator/(T value,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_r);
    return (vec_return /= value);
  }
  //      （        ） + - * /
  _IS_CUDA_CODE_ inline friend Vector3<T> operator+(const Vector3<T> &vec_l,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_l);
    return (vec_return += vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator-(const Vector3<T> &vec_l,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_l);
    return (vec_return -= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator*(const Vector3<T> &vec_l,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_l);
    return (vec_return *= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector3<T> operator/(const Vector3<T> &vec_l,
                                                    const Vector3<T> &vec_r) {
    Vector3<T> vec_return(vec_l);
    return (vec_return /= vec_r);
  }

  //
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector3<T> &vec_l,
                                               const T &value) {
    return (vec_l.x == value) && (vec_l.y == value) && (vec_l.z == value);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector3<T> &vec_l,
                                               const T &value) {
    return (vec_l.x != value) || (vec_l.y != value) || (vec_l.z != value);
  }
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector3<T> &vec_l,
                                               const Vector3<T> &vec_r) {
    return (vec_l.x == vec_r.x) && (vec_l.y == vec_r.y) && (vec_l.z == vec_r.z);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector3<T> &vec_l,
                                               const Vector3<T> &vec_r) {
    return (vec_l.x != vec_r.x) || (vec_l.y != vec_r.y) || (vec_l.z != vec_r.z);
  }

  //
  _IS_CUDA_CODE_ inline friend std::ostream &operator<<(std::ostream &os,
                                                        const Vector3<T> &vec) {
    os << vec.x << ", " << vec.y << ", " << vec.z;
    return os;
  }
};
//! Vector-4 template class
template <typename T>
class Vector4 : public Vector4_<T> {
 public:
  //
  _IS_CUDA_CODE_ Vector4(){};
  // explicit
  _IS_CUDA_CODE_ explicit Vector4(const T &value) {
    this->x = value;
    this->y = value;
    this->z = value;
    this->w = value;
  }
  _IS_CUDA_CODE_ explicit Vector4(const T *data) {
    this->x = data[0];
    this->y = data[1];
    this->z = data[2];
    this->w = data[3];
  }
  _IS_CUDA_CODE_ Vector4(const T v0, const T v1, const T v2, const T v3) {
    this->x = v0;
    this->y = v1;
    this->z = v2;
    this->w = v2;
  }
  _IS_CUDA_CODE_ Vector4(const Vector4_<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = vec.w;
  }

  //
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //
  _IS_CUDA_CODE_ Vector3<T> &operator=(const T value) {
    this->x = value;
    this->y = value;
    this->z = value;
    this->w = value;
  }
  _IS_CUDA_CODE_ Vector3<T> &operator=(const Vector3<T> &vec) {
    this->x = vec.x;
    this->y = vec.y;
    this->z = vec.z;
    this->w = vec.w;
    return (*this);
  }

  // norm
  _IS_CUDA_CODE_ inline T norm() {
    T norm_len = (T)sqrt(this->x * this->x + this->y * this->y +
                         this->z * this->z + this->w * this->w);
    return norm_len;
  }
  // inner product
  _IS_CUDA_CODE_ inline T dot(const Vector4<T> &vec) {
    T inner_product = (T)(this->x * vec.x + this->y * vec.y + this->z * vec.z +
                          this->w * vec.w);
    return inner_product;
  }
  // normalize
  _IS_CUDA_CODE_ inline void normlize() {
    T norm_len = this->norm();
    this->x /= norm_len;
    this->y /= norm_len;
    this->z /= norm_len;
    this->w /= norm_len;
  }

  //      （    ，     ！）
  //    += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator+=(Vector4<T> &vec,
                                                      T value) {
    vec.x += value;
    vec.y += value;
    vec.z += value;
    vec.w += value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator-=(Vector4<T> &vec,
                                                      T value) {
    vec.x -= value;
    vec.y -= value;
    vec.z -= value;
    vec.w -= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator*=(Vector4<T> &vec,
                                                      T value) {
    vec.x *= value;
    vec.y *= value;
    vec.z *= value;
    vec.w *= value;
    return vec;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator/=(Vector4<T> &vec,
                                                      T value) {
    vec.x /= value;
    vec.y /= value;
    vec.z /= value;
    vec.w /= value;
    return vec;
  }
  //      （        ） += -= *= /=
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator+=(Vector4<T> &vec_l,
                                                      const Vector4<T> &vec_r) {
    vec_l.x += vec_r.x;
    vec_l.y += vec_r.y;
    vec_l.z += vec_r.z;
    vec_l.w += vec_r.w;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator-=(Vector4<T> &vec_l,
                                                      const Vector4<T> &vec_r) {
    vec_l.x -= vec_r.x;
    vec_l.y -= vec_r.y;
    vec_l.z -= vec_r.z;
    vec_l.w -= vec_r.w;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator*=(Vector4<T> &vec_l,
                                                      const Vector4<T> &vec_r) {
    vec_l.x *= vec_r.x;
    vec_l.y *= vec_r.y;
    vec_l.z *= vec_r.z;
    vec_l.w *= vec_r.w;
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> &operator/=(Vector4<T> &vec_l,
                                                      const Vector4<T> &vec_r) {
    vec_l.x /= vec_r.x;
    vec_l.y /= vec_r.y;
    vec_l.z /= vec_r.z;
    vec_l.w /= vec_r.w;
    return vec_l;
  }
  //    + - * /
  _IS_CUDA_CODE_ inline friend Vector4<T> operator+(const Vector4<T> &vec_l,
                                                    T value) {
    Vector4<T> vec_return(vec_l);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator+(T value,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_r);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator-(const Vector4<T> &vec_l,
                                                    T value) {
    Vector4<T> vec_return(vec_l);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator-(T value,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_r);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator*(const Vector4<T> &vec_l,
                                                    T value) {
    Vector4<T> vec_return(vec_l);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator*(T value,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_r);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator/(const Vector4<T> &vec_l,
                                                    T value) {
    Vector4<T> vec_return(vec_l);
    return (vec_return /= value);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator/(T value,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_r);
    return (vec_return /= value);
  }
  //      （        ） + - * /
  _IS_CUDA_CODE_ inline friend Vector4<T> operator+(const Vector4<T> &vec_l,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_l);
    return (vec_return += vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator-(const Vector4<T> &vec_l,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_l);
    return (vec_return -= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator*(const Vector4<T> &vec_l,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_l);
    return (vec_return *= vec_r);
  }
  _IS_CUDA_CODE_ inline friend Vector4<T> operator/(const Vector4<T> &vec_l,
                                                    const Vector4<T> &vec_r) {
    Vector4<T> vec_return(vec_l);
    return (vec_return /= vec_r);
  }

  //
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector4<T> &vec_l,
                                               const T &value) {
    return (vec_l.x == value) && (vec_l.y == value) && (vec_l.z == value) &&
           (vec_l.w == value);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector4<T> &vec_l,
                                               const T &value) {
    return (vec_l.x != value) || (vec_l.y != value) || (vec_l.z != value) ||
           (vec_l.w != value);
  }
  _IS_CUDA_CODE_ inline friend bool operator==(const Vector4<T> &vec_l,
                                               const Vector4<T> &vec_r) {
    return (vec_l.x == vec_r.x) && (vec_l.y == vec_r.y) &&
           (vec_l.z == vec_r.z) && (vec_l.w == vec_r.w);
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(const Vector4<T> &vec_l,
                                               const Vector4<T> &vec_r) {
    return (vec_l.x != vec_r.x) || (vec_l.y != vec_r.y) ||
           (vec_l.z != vec_r.z) || (vec_l.w != vec_r.w);
  }

  //
  _IS_CUDA_CODE_ inline friend std::ostream &operator<<(std::ostream &os,
                                                        const Vector4<T> &vec) {
    os << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w;
    return os;
  }
};

//   -X    （  ，     ，            ！）
template <typename T, int Length>
class VectorX : public VectorX_<T, Length> {
 public:
  //
  _IS_CUDA_CODE_ VectorX(){};
  // explicit
  _IS_CUDA_CODE_ explicit VectorX(const T &value) {
    for (int i = 0; i < Length; i++) {
      this->data[i] = value;
    }
  }
  _IS_CUDA_CODE_ explicit VectorX(const T *data) {
    for (int i = 0; i < Length; i++) {
      this->data[i] = data[i];
    }
  }
  _IS_CUDA_CODE_ VectorX(const VectorX_<T, Length> &vec) {
    for (int i = 0; i < Length; i++) {
      this->data[i] = vec->data[i];
    }
  }

  //
  _IS_CUDA_CODE_ T &operator[](int i) { return this->data[i]; }
  _IS_CUDA_CODE_ const T &operator[](int i) const { return this->data[i]; }

  //      （    ，     ！）
  //    += -= *= /=
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator+=(
      VectorX<T, Length> &vec, T value) {
    for (int i = 0; i < Length; i++) {
      vec[i] += value;
    }
    return vec;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator-=(
      VectorX<T, Length> &vec, T value) {
    for (int i = 0; i < Length; i++) {
      vec[i] -= value;
    }
    return vec;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator*=(
      VectorX<T, Length> &vec, T value) {
    for (int i = 0; i < Length; i++) {
      vec[i] *= value;
    }
    return vec;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator/=(
      VectorX<T, Length> &vec, T value) {
    for (int i = 0; i < Length; i++) {
      vec[i] /= value;
    }
    return vec;
  }
  //      （        ） += -= *= /=
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator+=(
      VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    for (int i = 0; i < Length; i++) {
      vec_l[i] += vec_r[i];
    }
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator-=(
      VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    for (int i = 0; i < Length; i++) {
      vec_l[i] -= vec_r[i];
    }
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator*=(
      VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    for (int i = 0; i < Length; i++) {
      vec_l[i] *= vec_r[i];
    }
    return vec_l;
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> &operator/=(
      VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    for (int i = 0; i < Length; i++) {
      vec_l[i] /= vec_r[i];
    }
    return vec_l;
  }
  //    + - * /
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator+(
      const VectorX<T, Length> &vec_l, T value) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator+(
      T value, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_r);
    return (vec_return += value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator-(
      const VectorX<T, Length> &vec_l, T value) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator-(
      T value, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_r);
    return (vec_return -= value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator*(
      const VectorX<T, Length> &vec_l, T value) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator*(
      T value, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_r);
    return (vec_return *= value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator/(
      const VectorX<T, Length> &vec_l, T value) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return /= value);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator/(
      T value, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_r);
    return (vec_return /= value);
  }
  //      （        ） + - * /
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator+(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return += vec_r);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator-(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return -= vec_r);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator*(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return *= vec_r);
  }
  _IS_CUDA_CODE_ inline friend VectorX<T, Length> operator/(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    VectorX<T, Length> vec_return(vec_l);
    return (vec_return /= vec_r);
  }

  //
  _IS_CUDA_CODE_ inline friend bool operator==(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    bool flag = true;
    for (int i = 0; i < Length; i++) {
      if (vec_l[i] != vec_r[i]) {
        flag = false;
        break;
      }
    }
    return flag;
  }
  _IS_CUDA_CODE_ inline friend bool operator!=(
      const VectorX<T, Length> &vec_l, const VectorX<T, Length> &vec_r) {
    bool flag = false;
    for (int i = 0; i < Length; i++) {
      if (vec_l[i] != vec_r[i]) {
        flag = true;
        break;
      }
    }
    return flag;
  }

  //
  _IS_CUDA_CODE_ inline friend std::ostream &operator<<(
      std::ostream &os, const VectorX<T, Length> &vec) {
    for (int i = 0; i < Length; i++) {
      os << vec[i] << ", ";
    }
    return os;
  }
};

// Reference: Eigen 3.2
#define MAKE_TYPES(Type, TypeSuffix)         \
  typedef Vector2<Type> Vector2##TypeSuffix; \
  typedef Vector3<Type> Vector3##TypeSuffix; \
  typedef Vector4<Type> Vector4##TypeSuffix;

MAKE_TYPES(float, f);
MAKE_TYPES(double, d);
MAKE_TYPES(char, c);
MAKE_TYPES(unsigned char, uc);
MAKE_TYPES(short, s);
MAKE_TYPES(unsigned short, us);
MAKE_TYPES(int, i);
MAKE_TYPES(unsigned int, ui);

#undef MAKE_TYPES

typedef struct _Line_segment {
  My_Type::Vector3f origin;
  My_Type::Vector3f dst;
} Line_segment;
}  // namespace My_Type
