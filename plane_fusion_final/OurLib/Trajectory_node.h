#pragma once

//! C/C++ IO
#include <cstdio>
#include <iostream>
#include <vector>

//! Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

//! my quaternons
#include "OurLib/My_quaternions.h"
#include "OurLib/My_quaternions_interface.h"

/**
 * @brief A templete class define trajectory node
 */
class Trajectory_node {
 public:
  /**
   * @brief Frame's time stamp or frame's index.
   */
  double time;
  /**
   * @brief The first element of translation.
   */
  float tx;
  /**
   * @brief The second element of translation.
   */
  float ty;
  /**
   * @brief The third element of translation.
   */
  float tz;
  /**
   * @brief quaternions represent the rotation.
   */
  My_Type::My_quaternionsf quaternions;


  /**
   * @brief Default constructor.
   */
  Trajectory_node();
  /**
   * @brief Constructor from translation and rotation expressed by quaternion.
   * @param timestamp Refer to timestamp.
   * @param _tx The first element of translation.
   * @param _ty The second element of translation.
   * @param _tz The third element of translation.
   * @param _qw Imaginary part of quaternion.
   * @param _qx The first element of real part of quaternion.
   * @param _qy The second element of real part of quaternion.
   * @param _qz The third element of real part of quaternion.
   */
  Trajectory_node(double timestamp, float _tx, float _ty, float _tz, float _qw,
                  float _qx, float _qy, float _qz);
  /**
   * @brief Constructor from translation and rotation expressed by 4x4 pose matrix.
   * @param time Timestamp
   * @param pose_mat 4x4 pose matrix.
   */
  Trajectory_node(double time, Eigen::Matrix4f pose_mat);
  /**
   * @brief Default destructor.
   */
  ~Trajectory_node();


//          \note	Tips: '=','()','[]','->' must override in the class.
//                          Never override them as friend functions!
  /**
   * @brief Overload assignment operator '='.
   * @param _node Assigned value.
   * @return Trajectory_node reference.
   */
  Trajectory_node &operator=(const Trajectory_node &_node) {
    if (this != &_node) {
      this->time = _node.time;
      this->tx = _node.tx;
      this->ty = _node.ty;
      this->tz = _node.tz;

      this->quaternions.qr = _node.quaternions.qr;
      this->quaternions.qx = _node.quaternions.qx;
      this->quaternions.qy = _node.quaternions.qy;
      this->quaternions.qz = _node.quaternions.qz;
    }
    return (*this);
  }

  /**
   * @brief Overload std::ostream operator <<.
   * @param out_stream std::ostream.
   * @param _node Trajectory_node to be printed.
   * @return std::ostream.
   */
  friend std::ostream &operator<<(std::ostream &out_stream,
                                  const Trajectory_node &_node);
};

/**
 * @brief Define Trajectory by containers filled with Trajectory_node.
 */
typedef std::vector<Trajectory_node> Trajectory;
