//-- #version 460 core
//-- #extension GL_ARB_shading_language_include : require
//-- #extension GL_NV_gpu_shader5 : enable
//-- #extension GL_NV_shader_buffer_load : enable

#include "./common/Uniforms.h"

// Vertex attributes
layout(location = 0) ___in vec4 in_position;
layout(location = 1) ___in vec4 in_normal;
layout(location = 2) ___in vec4 in_color;

___out vec3 vs_fragPos;
___out vec3 vs_normal;
___out vec4 vs_color;

void main(void) {
    vs_fragPos = in_position.xyz;
    vs_normal = in_normal.xyz;
    vs_color = in_color;
    gl_Position = uniforms.projMat * uniforms.viewMat * in_position;
}
