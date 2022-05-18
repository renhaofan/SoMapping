
//
////Windows Header
//#include <windows.h>
// GL
#include <GL/glew.h>
// freeglut
#include <GL/freeglut.h>
#include <GL/glut.h>

//
#include "My_matrix.h"
#include "My_vector.h"

//
void caculate_cube_vertex(My_Type::Vector3f *points, float width, float height,
                          float depth) {
  //
  points[0].x = width * 0.5;
  points[0].y = height * 0.5;
  points[0].z = depth * 0.5;

  points[1].x = width * 0.5;
  points[1].y = height * 0.5;
  points[1].z = -depth * 0.5;

  points[2].x = -width * 0.5;
  points[2].y = height * 0.5;
  points[2].z = -depth * 0.5;

  points[3].x = -width * 0.5;
  points[3].y = height * 0.5;
  points[3].z = depth * 0.5;
  //
  points[4].x = width * 0.5;
  points[4].y = -height * 0.5;
  points[4].z = depth * 0.5;

  points[5].x = width * 0.5;
  points[5].y = -height * 0.5;
  points[5].z = -depth * 0.5;

  points[6].x = -width * 0.5;
  points[6].y = -height * 0.5;
  points[6].z = -depth * 0.5;

  points[7].x = -width * 0.5;
  points[7].y = -height * 0.5;
  points[7].z = depth * 0.5;
}

//
// side_flag:         true=  ， false-
void draw_cube_surface(float width, float height, float depth, bool side_flag) {
  My_Type::Vector3f points[8];

  caculate_cube_vertex(points, width, height, depth);

  glEnable(GL_CULL_FACE); //

  if (side_flag) {
    glBegin(GL_QUADS);
    //
    glVertex3f(points[0].x, points[0].y, points[0].z);
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    glVertex3f(points[3].x, points[3].y, points[3].z);
    //
    glVertex3f(points[4].x, points[4].y, points[4].z);
    glVertex3f(points[7].x, points[7].y, points[7].z);
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[5].x, points[5].y, points[5].z);
    //
    glVertex3f(points[0].x, points[0].y, points[0].z);
    glVertex3f(points[3].x, points[3].y, points[3].z);
    glVertex3f(points[7].x, points[7].y, points[7].z);
    glVertex3f(points[4].x, points[4].y, points[4].z);
    //
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glVertex3f(points[5].x, points[5].y, points[5].z);
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    //
    glVertex3f(points[3].x, points[3].y, points[3].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[7].x, points[7].y, points[7].z);
    //
    glVertex3f(points[0].x, points[0].y, points[0].z);
    glVertex3f(points[4].x, points[4].y, points[4].z);
    glVertex3f(points[5].x, points[5].y, points[5].z);
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glEnd();
  } else {
    glBegin(GL_QUADS);
    //
    glVertex3f(points[3].x, points[3].y, points[3].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glVertex3f(points[0].x, points[0].y, points[0].z);
    //
    glVertex3f(points[5].x, points[5].y, points[5].z);
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[7].x, points[7].y, points[7].z);
    glVertex3f(points[4].x, points[4].y, points[4].z);
    //
    glVertex3f(points[4].x, points[4].y, points[4].z);
    glVertex3f(points[7].x, points[7].y, points[7].z);
    glVertex3f(points[3].x, points[3].y, points[3].z);
    glVertex3f(points[0].x, points[0].y, points[0].z);
    //
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[5].x, points[5].y, points[5].z);
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    //
    glVertex3f(points[7].x, points[7].y, points[7].z);
    glVertex3f(points[6].x, points[6].y, points[6].z);
    glVertex3f(points[2].x, points[2].y, points[2].z);
    glVertex3f(points[3].x, points[3].y, points[3].z);
    //
    glVertex3f(points[1].x, points[1].y, points[1].z);
    glVertex3f(points[5].x, points[5].y, points[5].z);
    glVertex3f(points[4].x, points[4].y, points[4].z);
    glVertex3f(points[0].x, points[0].y, points[0].z);
    glEnd();
  }

  glDisable(GL_CULL_FACE);
}

//
void draw_cube_line(float width, float height, float depth, float line_width) {
  My_Type::Vector3f points[8];

  caculate_cube_vertex(points, width, height, depth);

  glLineWidth(line_width);

  glBegin(GL_LINES);
  // X
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(points[5].x, points[5].y, points[5].z);
  glVertex3f(points[6].x, points[6].y, points[6].z);
  glVertex3f(points[4].x, points[4].y, points[4].z);
  glVertex3f(points[7].x, points[7].y, points[7].z);
  // Y
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[4].x, points[4].y, points[4].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(points[5].x, points[5].y, points[5].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(points[6].x, points[6].y, points[6].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glVertex3f(points[7].x, points[7].y, points[7].z);
  // Z
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(points[4].x, points[4].y, points[4].z);
  glVertex3f(points[5].x, points[5].y, points[5].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(points[7].x, points[7].y, points[7].z);
  glVertex3f(points[6].x, points[6].y, points[6].z);

  glEnd();
}

//   GL   （   ）
void draw_coordinate_GL(float length, float line_width) {
  My_Type::Vector3f points[7];

  //
  points[0].x = 0;
  points[0].y = 0;
  points[0].z = 0;

  // +X +Y +Z
  points[1].x = length;
  points[1].y = 0;
  points[1].z = 0;
  points[2].x = 0;
  points[2].y = length;
  points[2].z = 0;
  points[3].x = 0;
  points[3].y = 0;
  points[3].z = length;
  // -X -Y -Z
  points[4].x = -length;
  points[4].y = 0;
  points[4].z = 0;
  points[5].x = 0;
  points[5].y = -length;
  points[5].z = 0;
  points[6].x = 0;
  points[6].y = 0;
  points[6].z = -length;

  //
  glLineWidth(line_width * 2.0);
  glColor4f(1.0, 0.0, 0.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glEnd();
  glColor4f(0.0, 1.0, 0.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glEnd();
  glColor4f(0.0, 0.0, 1.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glEnd();

  //
  glLineWidth(line_width);
  glColor4f(1.0, 0.0, 0.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[4].x, points[4].y, points[4].z);
  glEnd();
  glTranslatef(-points[4].x, -points[4].y, -points[4].z);
  glRotatef(90.0, 0.0, 1.0, 0.0);
  glutSolidCone(length * 0.05, length * 0.2, 10, 10);
  glRotatef(-90.0, 0.0, 1.0, 0.0);
  glTranslatef(points[4].x, points[4].y, points[4].z);

  glColor4f(0.0, 1.0, 0.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[5].x, points[5].y, points[5].z);
  glEnd();
  glTranslatef(-points[5].x, -points[5].y, -points[5].z);
  glRotatef(-90.0, 1.0, 0.0, 0.0);
  glutSolidCone(length * 0.05, length * 0.2, 10, 10);
  glRotatef(90.0, 1.0, 0.0, 0.0);
  glTranslatef(points[5].x, points[5].y, points[5].z);

  glColor4f(0.0, 0.0, 1.0, 0.7);
  glBegin(GL_LINES);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[6].x, points[6].y, points[6].z);
  glEnd();
  glTranslatef(-points[6].x, -points[6].y, -points[6].z);
  glutSolidCone(length * 0.05, length * 0.2, 10, 10);
  glTranslatef(points[6].x, points[6].y, points[6].z);

  //
  glLineWidth(1.0f);
}

// Draw camera Keyframe
void draw_keyframe_pose(float rect_width, float aspect_ratio,
                        float dapth_ratio) {
  My_Type::Vector3f points[4];
  float half_width = rect_width / 2;
  points[0].x = half_width;
  points[0].y = aspect_ratio * half_width;
  points[0].z = dapth_ratio * half_width;
  points[1].x = -points[0].x;
  points[1].y = +points[0].y;
  points[1].z = points[0].z;
  points[2].x = -points[0].x;
  points[2].y = -points[0].y;
  points[2].z = points[0].z;
  points[3].x = +points[0].x;
  points[3].y = -points[0].y;
  points[3].z = points[0].z;

  glBegin(GL_LINES);

  // Draw rectange
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glVertex3f(points[3].x, points[3].y, points[3].z);
  glVertex3f(points[0].x, points[0].y, points[0].z);

  // Draw lines
  glVertex3f(0, 0, 0);
  glVertex3f(points[0].x, points[0].y, points[0].z);
  glVertex3f(0, 0, 0);
  glVertex3f(points[1].x, points[1].y, points[1].z);
  glVertex3f(0, 0, 0);
  glVertex3f(points[2].x, points[2].y, points[2].z);
  glVertex3f(0, 0, 0);
  glVertex3f(points[3].x, points[3].y, points[3].z);

  glEnd();
}
