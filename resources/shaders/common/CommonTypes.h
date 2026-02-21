#ifndef COMMONTYPES_H
#define COMMONTYPES_H

#include "GLSLDefines.h"

struct Uniforms {
    mat4 viewMat;
    mat4 projMat;

    vec4 camera_pos;
    mat4 K;
    mat4 R;
    vec4 T;

    int num_gaussians;
    float near_plane;
    float far_plane;
    float scale_modifier;

    float SDF_scale;
    int selected_gaussian;
    float min_opacity;
    float padding1;

    float width;
    float height;
    float focal_x;
    float focal_y;

    int antialiasing;
    int front_to_back;
    int padding2;  // Add padding for 8-byte alignment
    int padding3;  // Add padding for 8-byte alignment

    vec4* restrict positions;
    vec4* restrict normals;
    vec4* restrict covX;
    vec4* restrict covY;
    vec4* restrict covZ;
    float* restrict sdf;
    float* restrict scale_neus;
    float* restrict sh_coeffs_red;
    float* restrict sh_coeffs_green;
    float* restrict sh_coeffs_blue;

    int* restrict visible_gaussians_counter;
    float* restrict gaussians_depth;
    int* restrict gaussians_indices;
    float* restrict sorted_depths;
    int* restrict sorted_gaussian_indices;

    vec4* restrict bounding_boxes;
    vec4* restrict conic_opacity;
    vec2* restrict eigen_vecs;
    vec4* restrict predicted_colors;

};

#endif //COMMONTYPES_H