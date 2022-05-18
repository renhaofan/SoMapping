/*!

	\file		my_math_functions.h

	\brief		Some auxiliary math functions

	\details

	\author		GongBingjian

	\date		2018-06-01

	\version	V2.0

	\par	Copyright (c):
	2018-2019 GongBingjian All rights reserved.

	\par	history
	2019-03-09 15:44:13	Doc by Doxygen

*/



// int
//     
//! Floor by stride
/*!
	\param	value	The value need to be floored by stride.

	\param	stride	The stride value.

	\return	A int value which floored by stride.
	
	\note	For example, 42 floor by stride 5 is 45. And 43 floor by stride 5 is 40.
*/
int floor_by_stride(int value, int stride);
//     
//! Round by stride
/*!
	\see	Similar to floor_by_stride(int value, int stride)

	\note	For example, 42 round by stride 5 is 40. And 43 round by stride 5 is 45.
*/
int round_by_stride(int value, int stride);
//     
//! Ceil by stride
/*!
	\see	Similar to floor_by_stride(int value, int stride)

	\note	For example, 42 round by stride 5 is 45. And 43 round by stride 5 is 45.
*/
int ceil_by_stride(int value, int stride);


// float
//     
float floor_by_stride(float value, float stride);
//     
float round_by_stride(float value, float stride);
//     
float ceil_by_stride(float value, float stride);


// double
//     
double floor_by_stride(double value, double stride);
//     
double round_by_stride(double value, double stride);
//     
double ceil_by_stride(double value, double stride);



// float     
int compare_float(const void * a, const void * b);
//   4     float    
int compare_float_elm4_by4(const void * a, const void * b);
//   1     float    
int compare_float_elm3_by1(const void * a, const void * b);





