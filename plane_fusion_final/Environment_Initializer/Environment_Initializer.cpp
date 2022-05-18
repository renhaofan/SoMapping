//
#include "Environment_Initializer.h"

// C/C++ IO
#include <cstdio>
#include <iostream>
// Namespace
using namespace std;

// CUDA head files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Ceres head files
#include "ceres/ceres.h"
#include "ceres/rotation.h"

//!
#pragma comment( \
    lib, "C:/English_path/glog-0.4.0/glog-0.4.0-build/Release/glog.lib")
//#pragma comment(lib,"glog.lib")

//
Environment_Initializer::Environment_Initializer() {}
Environment_Initializer::~Environment_Initializer() {}
//
Environment_Initializer::Environment_Initializer(bool print_detail) {
  this->print_detail_informations = print_detail;
}

//
void Environment_Initializer::init_environment(int argc, char **argv) {
  // Initiate CUDA device
#pragma region(Initiate CUDA device)

  // Reset GPU
  cudaDeviceReset();

  // CUDA device ID
  int devID;
  // CUDA device properties
  cudaDeviceProp deviceProps;

  // Find CUDA device
  devID = findCudaDevice(argc, (const char **)argv);

  // Gather GPU information
  // checkCudaErrors((CUresult)cudaGetDeviceProperties(&deviceProps, devID));
  // Store some device paramenters
  this->max_TperB = deviceProps.maxThreadsPerBlock;
  this->GPU_clock_rate = deviceProps.clockRate * 1000;
  // Show detail information
  if (this->print_detail_informations) {
    cout << "CUDA device ID = " << devID << endl;
    cout << "CUDA device is \t\t\t" << deviceProps.name << endl;
    cout << "CUDA max Thread per Block is \t" << deviceProps.maxThreadsPerBlock
         << endl;
    cout << "GPU clock frequency is \t\t" << deviceProps.clockRate << "\tkHz"
         << endl;
    // Max number of threads per multi-processor
    cout << "CUDA max Thread per SM is \t"
         << deviceProps.maxThreadsPerMultiProcessor << endl;
    cout << "CUDA Warp size is \t\t" << deviceProps.warpSize << endl;
    // number of multi-processors
    cout << "CUDA SM counter\t\t\t" << deviceProps.multiProcessorCount << endl;
    cout << "CUDA global memory size is \t"
         << deviceProps.totalGlobalMem / 1024 / 1024 << "\tMB" << endl;
    cout << "CUDA register per block is \t" << deviceProps.regsPerBlock << "\t"
         << endl;
    cout << "CUDA register per SM is \t" << deviceProps.regsPerMultiprocessor
         << "\t" << endl;
  }

#pragma endregion

  // Initiate google logging.
#pragma region(Initiate google logging)

  //#ifdef _LOGGING_H_
  google::InitGoogleLogging(argv[0]);
  //#endif

#pragma endregion
}
