#include "Trajectory_node.h"

//
Trajectory_node::Trajectory_node() {}
Trajectory_node::~Trajectory_node() {}

//
Trajectory_node::Trajectory_node(double timestamp, float _tx, float _ty,
                                 float _tz, float _qw, float _qx, float _qy,
                                 float _qz) {
  // Timestamp
  this->time = timestamp;
  // Translation
  this->tx = _tx;
  this->ty = _ty;
  this->tz = _tz;
  // Rotation represented in quaternions
  this->quaternions.qr = _qw;
  this->quaternions.qx = _qx;
  this->quaternions.qy = _qy;
  this->quaternions.qz = _qz;
}

//
Trajectory_node::Trajectory_node(double timestamp, Eigen::Matrix4f pose_mat) {
  // Timestamp
  this->time = timestamp;
  // Translation
  this->tx = pose_mat(0, 3);
  this->ty = pose_mat(1, 3);
  this->tz = pose_mat(2, 3);
  // Roatation
  Eigen::Matrix3f rot_mat = pose_mat.block(0, 0, 3, 3);
  my_convert(this->quaternions, rot_mat);

  //// Verification
  // Eigen::Quaternionf eigen_quater;
  // eigen_quater = rot_mat;
  // printf("%f, %f, %f, %f\r\n",
  //	   (float)this->quaternions.qr - (float)eigen_quater.w(), \
	//	   (float)this->quaternions.qx - (float)eigen_quater.x(), \
	//	   (float)this->quaternions.qy - (float)eigen_quater.y(), \
	//	   (float)this->quaternions.qz - (float)eigen_quater.z());
}

//
std::ostream &operator<<(std::ostream &out_stream,
                         const Trajectory_node &_node) {
  out_stream << _node.time << "\t" << _node.tx << "\t" << _node.ty << "\t"
             << _node.tz << "\t" << _node.quaternions;

  return out_stream;
};
