#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_sdf;
layout(location = 2) in vec3 in_dc;
layout(location = 3) in vec3 in_cov_row0;
layout(location = 4) in vec3 in_cov_row1;
layout(location = 5) in vec3 in_cov_row2;

out vec3 v_center;
out float v_opacity;
out vec3 v_dc;
out mat2 v_cov;

uniform mat4 projection;
uniform mat4 view;
uniform float GSize;

float sdfToOpacity(float sdf, float radius) {
    float x = abs(sdf);
    if (x > radius) return 0.0;
    x = x / radius;
    return exp(-x * x);  // Or use other falloff (see below)
}

void main() {
    vec4 cam_position = view * vec4(in_position, 1.0);
    gl_Position = projection * cam_position;

    // Convert clip space to NDC (Normalized Device Coordinates)
    vec3 ndc = gl_Position.xyz / gl_Position.w;
    
    // Map from NDC [-1, 1] to screen/image space [0, 1]
    v_center = ndc;//.xy; // * 0.5 + 0.5;

    v_opacity = sdfToOpacity(in_sdf, GSize); //20.0 * GSsize);

    v_dc = in_dc;

    /*mat3 cov = mat3(
        0.001, 0.0, 0.0,
        0.0, 0.001, 0.0,
        0.0, 0.0, 0.001
    );*/
    mat3 cov = mat3(in_cov_row0, in_cov_row1, in_cov_row2); // If row-major

    vec3 x = cam_position.xyz;

    // Extract rows of P
    //mat4 MVP3x3 = projection * view * model;
    vec4 p0 = projection[0];
    vec4 p1 = projection[1];
    vec4 p3 = projection[3];

    float w = dot(p3.xyz, x) + p3.w;
    float x_ndc = dot(p0.xyz, x) + p0.w;
    float y_ndc = dot(p1.xyz, x) + p1.w;

    vec3 dx = (p0.xyz / w) - (x_ndc / (w * w)) * p3.xyz;
    vec3 dy = (p1.xyz / w) - (y_ndc / (w * w)) * p3.xyz;

    mat2x3 J = mat2x3(dx, dy);

    // Matrix multiplication: J * cov3D
    mat2x3 JC;
    JC[0] = cov * J[0]; // first row of J * cov3D
    JC[1] = cov * J[1]; // second row of J * cov3D

    // Now compute JC * J^T → result is 2x2
    mat2 cov2D;
    cov2D[0][0] = dot(JC[0], J[0]);
    cov2D[0][1] = dot(JC[0], J[1]);
    cov2D[1][0] = dot(JC[1], J[0]);
    cov2D[1][1] = dot(JC[1], J[1]);

    v_cov = inverse(cov2D); // pass for fragment shader
}