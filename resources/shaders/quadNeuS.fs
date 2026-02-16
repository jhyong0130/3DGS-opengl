//-- #version 460 core
//-- #extension GL_ARB_shading_language_include :   require
//-- #extension GL_NV_gpu_shader5 : enable
//-- #extension GL_NV_shader_buffer_load : enable
//-- #extension GL_ARB_bindless_texture : enable

#include "./common/Uniforms.h"

___out vec4 out_Color;
//layout(location = 0) out vec4 accumColor; // accumulates alpha * color
//layout(location = 1) out vec4 accumAlpha; // accumulates alpha only

___flat ___in int InstanceID;
___in vec2 local_coord; // offset of the corner of the oriented bounding box from the center of the 2D ellipse, in pixels
___flat ___in vec2 center;
___flat ___in mat3 cov3D; //cov3D_inv; 

const float PI = 3.1415926535897932384626433832795;

float clip_sdf (float sdf) {
    return min(max(sdf, -30.0f), 30.0f);
}


float erfc_approx(float x)
{
    // constants from Abramowitz & Stegun (formula 7.1.26)
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.5 * z);

    float ans = t * exp(-z * z - 1.26551223 +
        t * ( 1.00002368 +
        t * ( 0.37409196 +
        t * ( 0.09678418 +
        t * (-0.18628806 +
        t * ( 0.27886807 +
        t * (-1.13520398 +
        t * ( 1.48851587 +
        t * (-0.82215223 +
        t * ( 0.17087277 ))))))))));
    return (x >= 0.0) ? ans : 2.0 - ans;
}

float erf(float x) {
    return 1.0 - erfc_approx(x);
}

float compute_alpha(mat3 cov3D_inv, vec3 cam, vec3 site, float sdf_site, vec3 normal, vec3 local_ray, float scale_neus) {
    vec3 local_delta = (cam - site);
    const float a = dot(local_ray, cov3D_inv * local_ray);
    const float b = dot(local_ray, cov3D_inv * local_delta);
    const float c = dot(local_delta, cov3D_inv * local_delta);
    const float lambda = dot(local_ray, normal) /1000.0f ;
    const float delta = sdf_site + dot(local_delta, normal) / 1000.0f;

    float bb = b/2.0f + scale_neus * lambda * delta;
    float cc = c/2.0f + scale_neus * delta * delta;
    float aa = a/2.0f + scale_neus * lambda*lambda;
    
    if (aa < 1e-6f) return 0.0f;
    float K = (sqrt(PI)/(2.0f*sqrt(aa))); // * sqrt(uniforms.scale_neus);
    float power = K * exp(bb*bb/aa - cc) * erfc_approx(bb/sqrt(aa)) * sqrt(aa/2.0f);
    return min(1.0f, power); //lambda < 0.0 ? min(1.0, power) : 0.0;
}

void main(void){

    const vec4 color = uniforms.predicted_colors[InstanceID];
    int Gaussian_ID = uniforms.sorted_gaussian_indices[InstanceID];
    const vec3 site = vec3(uniforms.positions[Gaussian_ID]);
    vec3 normal_world = vec3(uniforms.normals[Gaussian_ID]);
    const float sdf_site = 0.0f;//uniforms.sdf[Gaussian_ID];
       float scale_neus = uniforms.scale_neus[Gaussian_ID];

    vec2 curr_pix = local_coord + center;
    vec2 ndc = (curr_pix / vec2(uniforms.width, uniforms.height)) * 2.0 - 1.0;
    vec4 unproj = inverse(uniforms.projMat) * vec4(ndc, 1.0, 1.0);
    vec3 cam_ray_cam = normalize(unproj.xyz / unproj.w);
    
    // convert ray to world coords:
    mat3 R = mat3(uniforms.viewMat); // check convention: R should be world->camera 
    //vec3 cam_ray_world = transpose(R) * cam_ray_cam; // camera->world
    vec3 normal_cam = normalize(R * normal_world);
    vec3 mean = vec3(uniforms.viewMat * vec4(site, 1.0f));
    //vec3 mean = vec3(mat3(uniforms.R) * site + vec3(uniforms.T));
    //float dist = (proj_pixel.x-curr_pix.x) * (proj_pixel.x-curr_pix.x) + (proj_pixel.y-curr_pix.y) * (proj_pixel.y-curr_pix.y);
    
    // Compute inverse of cov3D
    mat3 cov3D_reg = cov3D;
    float det_cov = determinant(cov3D_reg);
    if (abs(det_cov) < 1.0e-30) {
        ___discard;
    }
    mat3 cov3D_inv = inverse(cov3D_reg);
    //mat3 cov3D_inv = mat3(0.01f); // debug alpha
    //vec3 cam =  vec3(uniforms.camera_pos);
    vec3 cam =  vec3(0.0f);
    //vec3 local_delta = (cam - mean);
    //const float lambda = dot(cam_ray_cam, normal_cam);
    //const float delta = sdf_site + dot(local_delta, normal_cam);
    //
    //if (any(isnan(cov3D_inv[0])) || any(isnan(cov3D_inv[1])) || any(isnan(cov3D_inv[2])) || 
    //     abs(lambda) < 1.0e-30 || lambda > 0.0f || delta < 0.0f) {
    //     ___discard;
    // }
 
    
   // 2D Analytical Splatting
   const vec4 conic_opacity = uniforms.conic_opacity[InstanceID];
   const mat2 conic = mat2(conic_opacity.x, conic_opacity.y,
                       conic_opacity.y, conic_opacity.z);
   const float opacity = conic_opacity.w;
   const float power = -0.5f * dot(local_coord, conic * local_coord);
   //if (power < -3.0f) {
   //    ___discard;
   //}
   float alpha_gs = min(0.99f, opacity * exp(power));
   
   // calculate final alpha
   //float alpha = compute_alpha(cov3D_inv, cam, mean, sdf_site, normal_cam, cam_ray_cam, scale_neus); //neuS
   float alpha = alpha_gs; 
    
   if (!isnan(alpha) && !isinf(alpha)) {
       out_Color = vec4(vec3(color) * alpha, alpha);
       //out_Color = vec4(vec3(a), 1.0f); // debug alpha
   }
}