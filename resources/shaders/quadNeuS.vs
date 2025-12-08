//-- #version 460 core
//-- #extension GL_ARB_shading_language_include :   require
//-- #extension GL_NV_gpu_shader5 : enable
//-- #extension GL_NV_shader_buffer_load : enable

#include "./common/Uniforms.h"
#include "./common/Covariance.h"

___flat ___out int InstanceID; // pass the index of the ellipse to the fragment shader
___out vec2 local_coord;
___flat ___out mat3 cov3D; //_inv;
___flat ___out vec2 center;

uniform int flipY;

void main(void){

    InstanceID = gl_VertexID / 6;

    // Corners:
    // 2 3
    // 0 1

    // indices:  0  1  2  3  2  1
    // x:       -1 +1 -1 +1 -1 +1
    // y:       -1 -1 +1 +1 +1 -1
    
    float x = (gl_VertexID % 2 == 0) ? -1.0f : +1.0f;
    float y = (((gl_VertexID+1) % 6) < 3) ? -1.0f : + 1.0f;
    const vec2 corner = vec2(x, y);
    
    int Gaussian_ID = uniforms.sorted_gaussian_indices[InstanceID];
    //const vec3 scale = vec3(uniforms.scales[Gaussian_ID]);
    const float scale_modifier = uniforms.scale_modifier;

    // covariance matrix in camera coordinate system
    //cov3D = computeCov3D(uniforms.covX[Gaussian_ID], uniforms.covY[Gaussian_ID], uniforms.covZ[Gaussian_ID], uniforms.rotations[Gaussian_ID], scale, scale_modifier, mat3(uniforms.R));
    cov3D = computeCov3D(uniforms.covX[Gaussian_ID], uniforms.covY[Gaussian_ID], uniforms.covZ[Gaussian_ID], scale_modifier, mat3(1.0f));

    // framebuffer size
    const float width = uniforms.width;
    const float height = uniforms.height;

    const vec4 box = uniforms.bounding_boxes[InstanceID]; // oriented bounding box
    center = vec2(box.x, box.y); // bounding box center, in pixels
    const vec2 half_extent = 2.0f*vec2(box.z, box.w); // bounding box half size, in pixels

    // direction of major axis
    const vec2 dir = uniforms.eigen_vecs[InstanceID];

    // ellipse rotation matrix in 2D
    const mat2 rotation = mat2(dir.x, dir.y, -dir.y, dir.x);

    // offset of the corner of the oriented bounding box from the center of the 2D ellipse, in pixels
    local_coord = rotation * (half_extent * corner);

    // normalized coordinates
    const vec2 ndc = (center + local_coord) / vec2(width, height) * 2.0f - 1.0f;
    gl_Position = vec4(ndc.x, flipY == 0 ? ndc.y : -ndc.y, 0.0f, 1.0f);
}