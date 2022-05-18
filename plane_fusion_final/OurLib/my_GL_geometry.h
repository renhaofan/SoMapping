#pragma once


//         
// side_flag:         true=  ， false-  
void draw_cube_surface(float width, float height, float depth, bool side_flag);

//        
void draw_cube_line(float width, float height, float depth, float line_width);

//   GL   
void draw_coordinate_GL(float length, float line_width);

// Draw camera Keyframe
void draw_keyframe_pose(float rect_width, float aspect_ratio, float dapth_ratio);

