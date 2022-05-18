

// C/C++ IO
#include <iostream>
#include <stdio.h>
using namespace std;

// OpenGL header files
// glew
#include <GL/glew.h>
// freeglut
#include <GL/freeglut.h>
#include <GL/glut.h>

// CUDA header files
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//
#include <cuda_gl_interop.h>

// OpenGL      CUDA
void GL_buffer_object_init_with_CUDA(GLuint *GL_BO,
                                     cudaGraphicsResource **CUDA_GR,
                                     const void *data, unsigned int size,
                                     GLenum GL_BO_catalogue,
                                     GLenum GL_copy_flag,
                                     unsigned int CUDA_flag);

// OpenGL error inspector
void checkGL(const string &func, const char *const file, int const line);
#define CheckGLError(val) checkGL((val), __FILE__, __LINE__)
