
#include "My_frame.h"
#include "GL/glew.h"

// OpenGL
void my_load_frame(My_frame<float> frame) {
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(frame.mat.data);
}

// OpenGL
void my_load_frame(My_frame<double> frame) {
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixd(frame.mat.data);
}
