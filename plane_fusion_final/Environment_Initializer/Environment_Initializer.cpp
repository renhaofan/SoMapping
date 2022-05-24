#include "Environment_Initializer.h"

#include <cstdio>
#include <iostream>
using namespace std;

// CUDA headers, for the CUDA runtime routines (prefixed with "cuda_").
#include <cuda.h>
#include <cuda_runtime.h>
// CUDA helper functions and utilities.
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Ceres head files
#include "ceres/ceres.h"
#include "ceres/rotation.h"

Environment_Initializer::Environment_Initializer() {}
Environment_Initializer::Environment_Initializer(bool print_detail) {
  this->print_detail_information = print_detail;
}
Environment_Initializer::~Environment_Initializer() {}

void Environment_Initializer::init_environment(int argc, char **argv) {
  // <-------- Check GPU device, CUDA environment
  //           and print concerned information.   ------->
  // Reset GPU
  cudaDeviceReset();

  // CUDA device ID
  int devID;

  // CUDA device properties
  cudaDeviceProp deviceProps;

  // Find CUDA device
  devID = findCudaDevice(argc, (const char **)argv);

  // Gather GPU information
  {
    cudaError_t err = cudaSuccess;
    err = cudaGetDeviceProperties(&deviceProps, devID);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "File %s, Line %d, Function %s()\n Failed to get device "
              "properties.(error code %s)!\n",
              __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  // Store some device paramenters
  this->max_TperB = deviceProps.maxThreadsPerBlock;
  this->GPU_clock_rate = deviceProps.clockRate * 1000;

  // Show detail information
  if (this->print_detail_information) {
    cout << "CUDA device ID = " << devID << endl;
    cout << "CUDA device is \t\t" << deviceProps.name << endl;
    cout << "CUDA max Thread per Block is \t" << deviceProps.maxThreadsPerBlock
         << endl;
    cout << "GPU clock frequency is \t" << deviceProps.clockRate << "\tkHz"
         << endl;
    // Max number of threads per multi-processor
    cout << "CUDA max Thread per SM is \t"
         << deviceProps.maxThreadsPerMultiProcessor << endl;
    cout << "CUDA Warp size is \t\t" << deviceProps.warpSize << endl;
    // number of multi-processors
    cout << "CUDA SM counter\t\t" << deviceProps.multiProcessorCount << endl;
    cout << "CUDA global memory size is \t"
         << deviceProps.totalGlobalMem / 1024 / 1024 << "\tMB" << endl;
    cout << "CUDA register per block is \t" << deviceProps.regsPerBlock << "\t"
         << endl;
    cout << "CUDA register per SM is \t" << deviceProps.regsPerMultiprocessor
         << "\t" << endl;
  }

  //<------- Initiate google logging ------->
  //#ifdef _LOGGING_H_
//  FLAGS_log_dir = "./log";
//  google::InitGoogleLogging(argv[0]);
//  std::cout << "FLAGS_log_dir: ";
//  std::cout << FLAGS_log_dir << std::endl;
  //#endif
}
