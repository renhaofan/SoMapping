
//
#include "math.h"
#include "stdio.h"
//
#include "My_quaternions.h"

using namespace My_Type;

//! Force compiler make instantation of My_quaternionsf and My_quaternionsd.
template class _IS_CUDA_CODE_ My_quaternions<float>;
template class _IS_CUDA_CODE_ My_quaternions<double>;


