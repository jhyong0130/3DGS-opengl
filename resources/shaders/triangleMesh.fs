//-- #version 460 core
//-- #extension GL_ARB_shading_language_include : require
//-- #extension GL_NV_gpu_shader5 : enable
//-- #extension GL_NV_shader_buffer_load : enable

#include "./common/Uniforms.h"

___in vec3 vs_fragPos;
___in vec3 vs_normal;
___in vec4 vs_color;

___out vec4 out_Color;

void main(void) {
    vec3 N = normalize(vs_normal);
    vec3 viewDir = normalize(uniforms.camera_pos.xyz - vs_fragPos);
    
    // Two-sided lighting
    if (dot(N, viewDir) < 0.0) {
        N = -N;
    }
    
    vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
    
    float ambient = 0.5;
    float diff = max(dot(N, lightDir), 0.0);
    
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(N, halfwayDir), 0.0), 32.0);
    
    float lighting = ambient + 0.6 * diff + 0.2 * spec;
    
    vec3 finalColor = vs_color.rgb * lighting;
    out_Color = vec4(finalColor, vs_color.a);
}
