#pragma once



// C/C++ IO
#include <cstdio>
#include <iostream>
using namespace std;

// OpenCV
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>


//! Image Loader
/*!

*/
class Image_loader
{
public:

	//! Enum record which kind of file can be loaded.
	/*!

	*/
	enum ImageLoaderMode
	{
		NO_DATA,
		WITH_DEPTH_ONLY,
		WITH_COLOR_AND_DEPTH,
		UNEQUAL_COLOR_AND_DEPTH_FRAMES,
	};
	//! Which kind of file can be loaded.
	ImageLoaderMode image_loader_mode;


	//! Enum of image loader state
	/*!

	*/
	enum ImageLoaderState
	{
		END_OF_DATA,
		WAIT_FOR_DATA,
		PREPARED_TO_LOAD,
	};
	//! Image loader state
	ImageLoaderState image_loader_state;



	//! Default constructor/destructor
	virtual ~Image_loader(){};


	//! Whether ready to load one frame.
	/*!
		
	*/
	virtual bool is_ready_to_load_next_frame() const = 0;


	//! Load one frame image
	/*!
		\param	timestamp	timestamp of captured frame

		\param	color_mat	Color image matrix. (cv::Mat)

		\param	depth_mat	Depth image matrix. (cv::Mat)

		\return	bool	true	-	succeed	load images
						false	-	failed to load images
	*/
	virtual bool load_next_frame(double & timestamp, cv::Mat & color_mat, cv::Mat & depth_mat) = 0;


	//! Get depth image size
	virtual void get_depth_image_size(int & width, int & height) const = 0;


	//! Get color image size
	virtual void get_color_image_size(int & width, int & height) const = 0;


	//!
	/*!
	
	*/
	//virtual Calibration_paramenters get_calibration_parameters() const = 0;

};



//! Blank image generator
/*!

*/
class Blank_image_loader : public Image_loader
{
public:

	// --- Interface :

	//!
	bool is_ready_to_load_next_frame() const override
	{	return false; }

	//! Load next frame
	bool load_next_frame(double & timestamp, cv::Mat & color_mat, cv::Mat & depth_mat) override
	{	return false; }

	//!
	void get_depth_image_size(int & width, int & height) const override
	{	width = 0;	height = 0; }

	//!
	void get_color_image_size(int & width, int & height) const override
	{	width = 0;	height = 0;}


	// --- Members :

	//! Default constructor/destructor
	Blank_image_loader();
	~Blank_image_loader();

};



//! Offline image loader
/*!

*/
class Offline_image_loader : public Image_loader
{
public:
    enum DatasetMode
    {
        ICL,
        TUM,
        MyZR300,
        MyD435i,
        MyAzureKinect,

    };
	// --- Interface :

	//!
	bool is_ready_to_load_next_frame() const override;

	//! Load next frame
	bool load_next_frame(double & timestamp, cv::Mat & color_mat, cv::Mat & depth_mat) override;

    void read_calibration_parameters(string cal);
    void detect_images(string associate, string colordir, string depthdir, int dm);
	// --- Members :

	//! Color image parameters
	int color_width, color_height, color_element_size, color_channel_num;
	//! Depth image parameters
	int depth_width, depth_height, depth_element_size, depth_channel_num;

	//! Color image path list
	vector<string> color_path_vector;
	//! Timestamp list of images
	vector<double> color_timestamp_vector;
	//! Depth image path list
	vector<string> depth_path_vector;
	//! Timestamp list of images
	vector<double> depth_timestamp_vector;
    vector<double> image_timestamp_vector;

	//!
	vector<int> color_index_vector;
	//!
	vector<int> depth_index_vector;

	//! Frame index
	size_t frame_index = 0;
	//! Number of frames
	size_t number_of_frames;


	//! Default constructor/destructor
	Offline_image_loader();
	~Offline_image_loader();
	//! Initialize parameters during construction
	//Offline_image_loader(string color_folder, string depth_loader);
    Offline_image_loader(string cal, string dir, int dm);

	//! Initiation
	/*!
	
	*/
	void init(string color_folder, string depth_folder);


	//! Jump to specific frame index
	/*!

	*/
	bool jump_to_specific_frame(int frame_id);


	//! Detect images.
	/*!
		\param	color_folder	The path of color images.

		\param	depth_folder	The path of depth images.

		\return	void
	*/
	void detect_images(string color_folder, string depth_folder);



	//! Load one frame image to determine parameters of image.
	/*!
		\return	void
	*/
	void read_image_parameters();


	//!
	void get_depth_image_size(int & width, int & height) const override
	{	width = this->depth_width;	height = this->depth_height;	}

	//!
	void get_color_image_size(int & width, int & height) const override
	{	width = this->color_width;	height = this->color_height;	}


	//!
	/*!
		\param	print_all_pathes	Flag to print all pathes of color and depth images

		\return	void
	*/
	void print_state(bool print_all_pathes = false) const;

    void _findclose(long long int file);
};