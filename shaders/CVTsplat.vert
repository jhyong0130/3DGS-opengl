#version 330 core
layout(location = 0) in vec3 in_position;
//layout(location = 1) in mat3 in_covariance;

uniform vec3 p_center;
uniform vec3 p_normal;
uniform vec3 p_u;
uniform vec3 p_v;

out vec3 v_center;
//out mat3 v_cov;

void main() {
    //vec3 clipPos = in_position - dot(in_position-p_center, p_normal)*p_normal;
    gl_Position = vec4(dot(in_position-p_center, p_u), dot(in_position-p_center, p_v), 0.0, 1.0);

    v_center = in_position; // (clipPos.xy / clipPos.w); // NDC center
    //v_cov = in_covariance; // pass for fragment shader
}
