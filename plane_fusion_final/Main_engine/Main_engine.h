#pragma once

// Head file of components
#include "Environment_Initializer/Environment_Initializer.h"
#include "Data_engine/Data_engine.h"
#include "SLAM_system/SLAM_system.h"
#include "UI_engine/UI_engine.h"




//!
/*!
	\note	Singleton object!
*/
class Main_engine
{
public:
	//! The pointer to this static object.
	static Main_engine * instance_ptr;
	//! Member function for instantiating this static object.
	static Main_engine * instance(void)
	{
		if (instance_ptr == nullptr)	instance_ptr = new Main_engine();
		return instance_ptr;
	}


	//! Environment initializer
	Environment_Initializer * environment_initializer;
	//! Data loader/writer
	Data_engine * data_engine;
	//! SLAM system
	SLAM_system * SLAM_system_ptr;
	//! OpenGL user interface
	UI_engine * UI_engine_ptr;
	//
	Render_engine * render_engine_ptr;

	//! Default constructor/destructor
	Main_engine();
	~Main_engine();


	//! Initiate all components
	/*!
		\param	argc					Input the frist argument of main function.

		\param	argv					Input the second argument of main function.

		\param	image_loader_ptr		The pointer of Image_loader. 

		\param	output_folder			Output folder.(output estimated trajectory, mesh, etc.)

		\param	_is_ICL_NUIM_dataset	Dataset coordinate mode.

		\return	void
	*/
	void init(int argc, char ** argv, Image_loader * image_loader_ptr, string output_folder = "./", bool _is_ICL_NUIM_dataset = false);
	

	//! Run main loop
	/*!
		\return	void

		\note	Run the main loop.
	*/
	void run();


};

