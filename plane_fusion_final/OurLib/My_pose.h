/*!

        \file		My_pose.h

        \brief		Defination of 3D object pose. See My_pose.

        \details	Include rotation and translation. \n
                                Define \f$ R \f$ is a rotation of 3 times 3
   matrix.\n Define \f$ t \f$ is a translation of 3 times 1 vector.\n The pose
   matrix is defined as \f$ \begin{bmatrix}
                                        R & t\\
                                        0 & 1
                                        \end{bmatrix}\f$


        \author		GongBingjian

        \date		2019-03-08

        \par	history
        2019-03-08 17:30:06		Doc by Doxygen

*/

#pragma once

// Eigen
#include <Eigen/Dense>
using namespace Eigen;

// Trajectory
#include "Trajectory_node.h"
//
#include "My_matrix.h"
#include "My_quaternions.h"
#include "My_quaternions_interface.h"

//! A class define camera pose
/*!
        \brief		This class depends on Eigen!

        \details	Include rotation and translation. \n
                                Define \f$ R \f$ is a rotation of 3 times 3
   matrix.\n Define \f$ t \f$ is a translation of 3 times 1 vector.\n The pose
   matrix is defined as \f$ \begin{bmatrix}
                                                                                                R & t\\
                                                                                                0 & 1
                                                                                                \end{bmatrix}\f$
*/
class My_pose {
public:
  /*!	Default constructor.	*/
  //! Malloc device(GPU) memory by CUDA API
  My_pose();
  /*!	Default destructor.	*/
  //! Free device(GPU) memory by CUDA API
  ~My_pose();

  // Pose
  //! Eigen matrix of the pose.
  /*!
          The matrix is saved as \f$ \begin{bmatrix}
                                                          R & t\\
                                                          0 & 1
                                                          \end{bmatrix}
                                                          =
                                                          \begin{bmatrix}
                                                          m_0 & m_4 & m_8 &
     m_{12}\\
                                                          m_1 & m_5 & m_9 &
     m_{13}\\
                                                          m_2 & m_6 & m_{10} &
     m_{14}\\ m_3 & m_7 & m_{11} & m_{15} \end{bmatrix}\f$. The \f$ m_i \f$ is
     the \f$ i^{th} \f$ element of mat.data().
  */
  Matrix4f mat;
  // CUDA pose array
  //! CUDA device pose matrix
  float *dev_mat;
  //! CUDA device pose inverse matrix
  float *dev_mat_inv;

  //! Compute pose matrix by Trajectory_node, which represent rotation as
  //! quaternions.
  /*!
          \param	trace	The translation and rotation(represent as
     quaternions).

          \return	void

          \see	Trajectory_node
  */
  void load_pose(Trajectory_node &camera_pose);

  //! Synchronize pose matrix form CPU memory to device(GPU) memory.
  void synchronize_to_GPU();

  //! Synchronize pose matrix form device(GPU) memory to CPU memory.
  void synchronize_from_GPU();

  //! Rotation and translation calculate the rigid body transformation
  //! independently of each other.
  /*!
          \param	pose_transformation		The rotation and
     translation.

          \return	void
  */
  void transformation_Rt_divided(My_pose &pose_transformation);
};
