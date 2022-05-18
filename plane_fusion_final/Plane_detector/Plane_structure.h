//! Header file for plane map data structure


#pragma once

#include "OurLib/My_matrix.h"

//
#define MIN_PLANE_DISTANCE				0.02
//#define HISTOGRAM_STEP					0.02
//#define HISTOGRAM_WIDTH					256		// 0.08 * 64 = 2.56 * 2 (-2.56 to +2.56)
#define HISTOGRAM_STEP					0.05
#define HISTOGRAM_WIDTH					128		// 0.08 * 64 = 2.56 * 2 (-2.56 to +2.56)
// 
#define MAX_CURRENT_PLANES		64
#define MAX_MODEL_PLANES		1024

//
#define MAX_HIST_NORMALS		1024

//
#define MAX_VALID_DEPTH_M		9.0



//
#define MIN_CELLS_OF_DIRECTION	55.0f
#define MIN_CELLS_OF_PLANE		50.0f
//
#define MIN_AREA_OF_DIRECTION	1.0f
#define MIN_AREA_OF_PLANE		0.5f
//
#define MIN_CELL_NORMAL_INNER_PRODUCT_VALUE	0.97

//   Patch  
// 8 x 4 byte        
typedef struct struct_Cell_info
{
	// Cell points normal vector
	float nx, ny, nz;
	// Cell points Pposition
	float x, y, z;
	// Patch 
	int plane_index;
	// Patch          
	int counter;
	// Area of this super pixel (m^2)
	float area;
	//
	bool is_valid_cell;

}Cell_info;



//        
typedef struct struct_Plane_info
{
	//
	float x, y, z;
	//              
	float nx, ny, nz, d;
	//     
	float weight;
	//
	int pixel_number;
	//          
	int plane_index;
	//      Patch  
	int cell_num;
	//
	float area;
	//    Fragment Map   Block  
	int block_num;
	//      
	bool is_valid;
	//     
	int global_index;

	// Constructor
	struct_Plane_info(){};
	struct_Plane_info(float _nx, float _ny, float _nz, float _weight, int _plane_index, int _cell_num, bool _is_valid, int _global_index) : \
		nx(_nx), ny(_ny), nz(_nz), weight(_weight), plane_index(_plane_index), cell_num(_cell_num), is_valid(_is_valid), global_index(_global_index) {}

}Plane_info;



//            
typedef struct struct_Hist_normal
{
	// 
	float nx, ny, nz;
	// Patch   
	int counter;
	//
	float weight;

}Hist_normal;



//
typedef struct struct_Plane_match_info
{
	//
	int current_plane_id;
	//
	int model_plane_id;
	// Patch
	float match_weight;

}Plane_match_info;


// Size of super pixel CUDA block width 
#define SUPER_PIXEL_BLOCK_WIDTH		16

//
typedef struct struct_Super_pixel
{
	// Center (x, y)
	union 
	{
		float center_data[2];
		struct { float cx, cy;};
	};
	// Super pixel normal vector
	union
	{
		float normal_data[3];
		struct { float nx, ny, nz; };
	};
	// Super pixel position
	union
	{
		float position_data[3];
		struct { float px, py, pz; };
	};
	// number of valid pixels
	int valid_pixel_number;
	//
	bool is_planar_cell;

}Super_pixel;



// Plane Hash entries
#define ORDERED_PLANE_TABLE_LENGTH	0x40000
#define ORDERED_PLANE_TABLE_MASK	0x3FFFF
// Excess table length
#define EXCESS_PLANE_TABLE_LENGTH	0x10000

// Hash   
typedef struct PlaneHashEntry
{
	//  
	int position[2];
	// offset < 0 collision  offset >= 0  collision
	int offset;		//    int
	// ptr < 0   ptr >= 0    
	int ptr;

}PlaneHashEntry;


// Plane pixel block width/size
#define PLANE_PIXEL_BLOCK_WIDTH		16
#define PLANE_PIXEL_BLOCK_SIZE		(PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_BLOCK_WIDTH)
#define TRACK_PLANE_THRESHOLD		0.6
// Pixel block number
#define PIXEL_BLOCK_NUM				0x10000

// Plane pixel/block size in meter
//#define PLANE_PIXEL_SIZE			0.005
#define PLANE_PIXEL_SIZE			0.004
#define HALF_PLANE_PIXEL_SIZE		PLANE_PIXEL_SIZE * 0.5
#define PLANE_PIXEL_BLOCK_WIDTH_M	(PLANE_PIXEL_BLOCK_WIDTH * PLANE_PIXEL_SIZE)
//
typedef struct struct_Plane_PIXEL
{
	// Diffirent distance to plane (positive: outside of surface; negative : inside of surface)
	float diff;
	// Label of plane (0 means not plane)
	int plane_label;

}Plane_pixel;


//
typedef struct struct_Plane_coordinate
{
	My_Type::Vector3f x_vec, y_vec, z_vec;

}Plane_coordinate;



