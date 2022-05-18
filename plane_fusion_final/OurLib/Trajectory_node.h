#pragma once


// C/C++ IO
#include <cstdio>
#include <iostream>
#include <vector>
//!

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
//!
#include "OurLib/My_quaternions.h"
#include "OurLib/My_quaternions_interface.h"



//! A templete class define trajectory node
/*!
	\brief		

	\details	
	
*/
class Trajectory_node
{
public:
	//! Frame's time stamp or frame's index
	double time;
	//! Translation
	float tx, ty, tz;
	//! Quaternions represent the rotation.
	My_Type::My_quaternionsf quaternions;

	//! Default constructor/destructor
	Trajectory_node();
	~Trajectory_node();


	//! Construct from timestamp & TxTyTz & Quaternions
	/*!
		
	*/
	Trajectory_node(double timestamp,
					float _tx, float _ty, float _tz,
					float _qw, float _qx, float _qy, float _qz);


	//! Construct from timestamp & PoseMat(Eigen 4x4)
	/*!
		
	*/
	Trajectory_node(double time, Eigen::Matrix4f pose_mat);


	//! override '='
	/*
		\note	Tips: '=','()','[]','->' must override in the class. 
				Never override them as friend functions!
	*/
	Trajectory_node & operator= (const Trajectory_node & _node)
	{
		if (this != &_node)
		{
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

	//! Output stream.
	/*!
	
	*/
	friend std::ostream & operator<< (std::ostream & out_stream, const Trajectory_node & _node);
	


};


// Define Trajectory 
typedef	std::vector<Trajectory_node>	Trajectory;


