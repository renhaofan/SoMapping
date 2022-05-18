

#include "my_GL_functions.h"

// OpenGL      CUDA
void GL_buffer_object_init_with_CUDA(GLuint *GL_BO,
                                     cudaGraphicsResource **CUDA_GR,
                                     const void *data, unsigned int size,
                                     GLenum GL_BO_catalogue,
                                     GLenum GL_copy_flag,
                                     unsigned int CUDA_flag) {
  //
  glBindBuffer(GL_BO_catalogue, *GL_BO);
  //
  glBufferData(GL_BO_catalogue, size, data, GL_DYNAMIC_COPY);
  //        （                ）
  glBindBuffer(GL_BO_catalogue, NULL);
  //   CUDA    OpenGL
  cudaGraphicsGLRegisterBuffer(CUDA_GR, *GL_BO, cudaGLMapFlagsNone);
}

// GL
void checkGL(const string &func, const char *const file, int const line) {
  GLenum err = GL_NO_ERROR;
  while ((err = glGetError()) != GL_NO_ERROR) {
    string errInfo;
    switch (err) {
      case GL_INVALID_ENUM:
        errInfo = "GL_INVALID_ENUM\0";
        break;
      case GL_INVALID_VALUE:
        errInfo = "GL_INVALID_VALUE\0";
        break;
      case GL_INVALID_OPERATION:
        errInfo = "GL_INVALID_OPERATION\0";
        break;
      case GL_OUT_OF_MEMORY:
        errInfo = "GL_OUT_OF_MEMORY\0";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        errInfo = "GL_INVALID_FRAMEBUFFER_OPERATION\0";
        break;
    }

    fprintf(stderr, "GL error at %s:%d detected error: %s at %s\n", file, line,
            errInfo.c_str(), func.c_str());
    exit(0);
  }
}
