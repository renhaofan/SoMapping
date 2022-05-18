#pragma once

#include <float.h>
#include <math.h>

#include "OurLib/My_matrix.h"
#include "OurLib/My_quaternions.h"
template <typename T>
class My_frame {
 public:
  My_Type::Matrix44<T> mat;
  T view_distance;

  My_frame() {
    this->mat.set_identity();

    view_distance = 0.0;
  }
  ~My_frame() {}

  //
  void set_Identity() {
    this->mat.set_identity();

    view_distance = 0.0;
  }

  void load_frame_mat(My_frame<T> frame_a) { this->mat = frame_a.mat; }

  //
  void load_frame_rot(My_frame<T> frame_a) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        this->mat(j, i) = frame_a.mat(j, i);
      }
    }
  }

  //
  void load_frame_transfer(My_frame<T> frame_a) {
    this->mat(0, 3) = frame_a.mat(0, 3);
    this->mat(1, 3) = frame_a.mat(1, 3);
    this->mat(2, 3) = frame_a.mat(2, 3);
  }

  //
  void load_quaternions(My_Type::My_quaternions<T> quater) {
    //
    this->mat.load_from_quaternions(quater);
  }

  //
  void print() {
    if ((strcmp(this->mat[0]).name(), "double")) {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          printf("%+1.3d,\t", this->mat[i * 4 + j]);
        }
        printf("\r\n");
      }
    } else if (strcmp((this->mat[0]).name(), "float")) {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          printf("%+1.3f, \t", this->mat[i * 4 + j]);
        }
        printf("\r\n");
      }
    }
    printf("\r\n");
  }

  //
  void multi_frame_mat_right(const My_frame<T> &frame_a) {
    this->mat = this->mat * frame_a.mat;
  }

  //
  void multi_frame_mat_left(My_frame<T> &frame_a) {
    //
    this->mat = frame_a.mat * this->mat;
  }

  //
  void rotate_inFrame(T alpha, T beta, T gamma) {
    My_Type::My_quaternions<T> W_x(0, 1, 0, 0), W_y(0, 0, 1, 0),
        W_z(0, 0, 0, 1);
    My_Type::My_quaternions<T> F_x, F_y, F_z;
    My_Type::My_quaternions<T> rot_FinW, d_rot_q;

    //
    rot_FinW = this->mat.convert_to_quaternions();

    //
    F_x = rot_FinW * W_x * rot_FinW.inverse();
    F_y = rot_FinW * W_y * rot_FinW.inverse();
    F_z = rot_FinW * W_z * rot_FinW.inverse();

    My_Type::My_quaternions<T> rot_Fx(
        cos(alpha * 0.5), W_x.qx * sin(alpha * 0.5), W_x.qy * sin(alpha * 0.5),
        W_x.qz * sin(alpha * 0.5));
    My_Type::My_quaternions<T> rot_Fy(cos(beta * 0.5), F_y.qx * sin(beta * 0.5),
                                      F_y.qy * sin(beta * 0.5),
                                      F_y.qz * sin(beta * 0.5));
    My_Type::My_quaternions<T> rot_Fz(
        cos(gamma * 0.5), W_z.qx * sin(gamma * 0.5), W_z.qy * sin(gamma * 0.5),
        W_z.qz * sin(gamma * 0.5));
    d_rot_q = rot_Fx * rot_Fy * rot_Fz;

    this->mat(2, 3) += this->view_distance;
    //
    My_frame frame_Rd;
    frame_Rd.load_quaternions(d_rot_q);

    //
    this->multi_frame_mat_left(frame_Rd);

    //
    this->mat(2, 3) -= this->view_distance;
  }

  //
  void transfer_inFrame(T dx, T dy, T dz) {
    this->mat(0, 3) += dx;
    this->mat(1, 3) += dy;
    this->mat(2, 3) += dz;
  }

  //
  void transfer_inWrold(T dx, T dy, T dz) {
    My_Type::My_quaternions<T> W_x(0, 1, 0, 0), W_y(0, 0, 1, 0),
        W_z(0, 0, 0, 1);
    My_Type::My_quaternions<T> F_x, F_y, F_z;
    My_Type::My_quaternions<T> rot_FinW;

    //
    rot_FinW = this->mat.convert_to_quaternions();

    //
    F_x = rot_FinW * W_x * rot_FinW.inverse();
    F_y = rot_FinW * W_y * rot_FinW.inverse();
    F_z = rot_FinW * W_z * rot_FinW.inverse();

    //
    this->mat(0, 3) += F_x.qx * dx + F_y.qx * dy + F_z.qx * dz;
    this->mat(1, 3) += F_x.qy * dx + F_y.qy * dy + F_z.qy * dz;
    this->mat(2, 3) += F_x.qz * dx + F_y.qz * dy + F_z.qz * dz;
  }
};

//   GL
void my_load_frame(My_frame<double> frame);
void my_load_frame(My_frame<float> frame);
