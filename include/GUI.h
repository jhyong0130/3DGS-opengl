#ifndef __GUI_H
#define __GUI_H

#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

/*** Camera variables for OpenGL ***/
float Znear = 0.0;
float Zfar = 10.0;
GLfloat light_pos[] = { 0.0, -10.0, -10.0 };
GLfloat ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
GLfloat diffuseLight[] = { 0.8f, 0.8f, 0.8 };
GLfloat specular[] = { 0.3f, 0.3f, 0.3f, 1.0f };

GLfloat mag_sdf[] = { 10.0f, 0.0f };
GLfloat pix[] = { 0.0f, 0.0f };

// angle of rotation for the camera direction
float anglex = 0.0f;
float angley = 0.0f;

// actual vector representing the camera's direction
float lx = 0.0f, ly = 0.0f, lz = 1.0f;
float lxStrap = -1.0f, lyStrap = 0.0f, lzStrap = 0.0f;

// XZ position of the camera
float x = 0.5f, y = 0.5f, z = -5.0f;

float r = -10.0f;

int axis = 0;
bool axis_flag = false;
bool fill_flag = true;
bool key_4_flag = false;

// the key states. These variables will be zero
//when no key is being presses
float deltaAnglex = 0.0f;
float deltaAngley = 0.0f;
float deltaMove = 0;
float deltaStrap = 0;
float xOrigin = -1.0;
float yOrigin = -1.0;

float my_count;
float fps;
int frame_idx = 499;
int increment = 1;
long prev_time = 0;
bool _play_back = false;
bool frame_flag = false;
bool Bframe_flag = false;
bool mesh_flag = false;
bool display_mesh = false;
bool Q_flag = false;
bool SDF_flag = true;
bool Vol_flag = false;
bool E_flag = false;
bool GRAD_flag = true;


glm::mat3 cam1_intrinsics;
glm::vec4 cam1_distortion_1;
glm::vec4 cam1_distortion_2;
float cam1_D2RGB[16] = { 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0 };
glm::vec2 cam1_size_RGB;
glm::mat3 cam1_RGB_intrinsics;
glm::vec4 cam1_RGB_distortion_1;
glm::vec4 cam1_RGB_distortion_2;


float XPlane_pose[16] = { 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0 };

float cam1_pose[16] = { 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0 };

float cam2_pose[16] = { 0.999969, -0.00778767, -0.0009467, 0.0,
                        0.00764499, 0.994429, -0.105132, 0.0,
                        0.00176016, 0.105122, 0.994458, 0.0,
                        -31.9544 / 1000.0, -1.81078 / 1000.0, 4.15482 / 1000.0, 1.0 };

bool _view_cam_1 = false;
bool _view_cam_2 = false;

float visual_hull[6] = { -1.6, -1.6, -1.6, 1.6, 1.6, 1.6 };


GLfloat vertices[] = {
    0.1f, -0.1f, 0.0f,
   -0.1f, -0.1f, 0.0f,
    0.0f,  0.1f, 0.0f
};

GLfloat v_colors[] = {
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f,  0.0f, 1.0f, 1.0f };


// An array of 3 vectors which represents 3 vertices
static const GLfloat g_vertex_buffer_data[] = {
    // positions          // colors                     // texture coords
    0.1f, -0.1f, 0.1f,      1.0f, 1.0f, 0.0f, 1.0f,     1.0, 0.0,
    0.1f, 0.1f, 0.1f,       1.0f, 1.0f, 0.0f, 1.0f,     1.0, 1.0,
    -0.1, -0.1, 0.1f,    1.0f, 1.0f, 0.0f, 1.0f,     0.0, 0.0,
    -0.1, 0.1, 0.1f,     1.0f, 1.0f, 0.0f, 1.0f,     0.0, 1.0,
    -0.1, -0.1, 0.1f,    1.0f, 1.0f, 0.0f, 1.0f,     0.0, 0.0,
    0.1, 0.1, 0.1f,     1.0f, 1.0f, 0.0f, 1.0f,      1.0, 1.0
};

// An array of 3 vectors which represents 3 vertices
static const GLfloat g_vertex_buffer_frustum[] = {
    // positions          // colors
   0.1f, -0.1f, 0.1f,      0.0f, 1.0f, 0.0f, 1.0f,
   0.1f, 0.1f, 0.1f,       0.0f, 1.0f, 0.0f, 1.0f,
   0.0f,  0.0f, 0.0f,    0.0f, 1.0f, 0.0f, 1.0f,
    0.1, 0.1, 0.1,     0.0f, 1.0f, 0.0f, 1.0f,
    -0.1, 0.1, 0.1,    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f,  0.0f, 0.0f,    0.0f, 1.0f, 0.0f, 1.0f,
    -0.1, 0.1, 0.1,     0.0f, 1.0f, 0.0f, 1.0f,
    -0.1, -0.1, 0.1,    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f,  0.0f, 0.0f,     0.0f, 1.0f, 0.0f, 1.0f,
    -0.1, -0.1, 0.1,     0.0f, 1.0f, 0.0f, 1.0f,
    0.1,-0.1, 0.1,    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f,  0.0f, 0.0f,     0.0f, 1.0f, 0.0f, 1.0f
};

float quadVerticesX[] = {
    // positions             // texture coords
    0.0f, 1.0f, -1.0f,       1.0, 0.0,
    0.0f, 1.0f, 1.0f,       1.0, 1.0,
    0.0f, -1.0, -1.0,      0.0, 0.0,
    0.0f, -1.0, 1.0,       0.0, 1.0,
    0.0f, -1.0, -1.0,      0.0, 0.0,
    0.0f, 1.0, 1.0,        1.0, 1.0
    };


float quadVerticesY[] = {
    // positions             // texture coords
    1.0f, 0.0f, -1.0f,       1.0, 0.0,
    1.0f, 0.0f, 1.0f,       1.0, 1.0,
    -1.0, 0.0f, -1.0,      0.0, 0.0,
    -1.0, 0.0f, 1.0,       0.0, 1.0,
    -1.0, 0.0f, -1.0,      0.0, 0.0,
    1.0, 0.0f, 1.0,        1.0, 1.0
    };

float quadVerticesZ[] = {
    // positions             // texture coords
    1.0f, -1.0f, 0.0f,        1.0, 0.0,
    1.0f, 1.0f, 0.0f,        1.0, 1.0,
    -1.0, -1.0, 0.0f,       0.0, 0.0,
    -1.0, 1.0, 0.0f,       0.0, 1.0,
    -1.0, -1.0, 0.0f,       0.0, 0.0,
    1.0, 1.0, 0.0f,         1.0, 1.0
    };

#endif
