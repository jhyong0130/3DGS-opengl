// Vertex shader (used with Transform Feedback)
#version 330 core
layout(location = 0) in uint inIndex;
out vec3 outPosition;
out float outAttributes;

uniform samplerBuffer vertices_in;
uniform samplerBuffer vertices;
uniform samplerBuffer attributes;

uniform float thresh_vals[2];

float SDF_Sphere(vec3 point, float R) {
    return length(point) - R;
}

float SDF_Torus(vec3 point, float R_max, float R_min) {
    float qx = sqrt(point.x * point.x + point.y * point.y) - R_max;
    float qz = point.z;
    return sqrt(qx * qx + qz * qz) - R_min;
}

float SDF_func(vec3 point) {
    //return SDF_Sphere(point, 0.5f);
    return SDF_Torus(point, 0.6f, 0.4f);
}

vec3 SDF_Grad_Sphere(vec3 point) {
    return normalize(point);
}

vec3 SDF_Grad_Torus(vec3 point, float R_max, float R_min) {
    float a = sqrt(point.x * point.x + point.y * point.y);
    float b = sqrt((a - R_max) * (a - R_max) + point.z * point.z);

    return vec3(((a - R_max) / b) * (point.x / a), ((a - R_max) / b) * (point.y / a), point.z / b);
}

vec3 SDF_Grad_func(vec3 point) {
    //return SDF_Grad_Sphere(point);
    return SDF_Grad_Torus(point, 0.6f, 0.4f);
}

vec4 Center(int myIndex)
{
    vec3 curr_point = texelFetch(vertices_in, myIndex).xyz;
    vec3 center_out = vec3(0.0f, 0.0f, 0.0f);
    float tot_area = 0.0f;
    vec4 curr_center;
    
    float flag = 0.0f;
    float curr_flag = 0.0f;
    // loop through all 20 tetrehedra that tesselate the sphere
    for (int j = 0; j < 20; j++) {
        curr_center = texelFetch(vertices, 20 * myIndex + j);
        curr_flag = texelFetch(attributes, 20 * myIndex + j).x;
        flag = flag == 0.0f ? curr_flag : flag;
        
        center_out = center_out + curr_center.xyz * curr_center.w;
        tot_area = tot_area + curr_center.w;
    }
    
    if (tot_area > 0.0f) {
        center_out = center_out / tot_area;
    }

    float sdf_in = SDF_func(curr_point);
    float sdf_out = SDF_func(center_out);
    float lambda = 0.5f;

    if ((sdf_in - thresh_vals[0]) * (sdf_out-thresh_vals[0]) <= 0.0f) {
        lambda = 0.5f * (abs(sdf_in - thresh_vals[0])/(abs(sdf_in - thresh_vals[0]) + abs(sdf_out - thresh_vals[0])));
        return vec4((1.0-lambda)*curr_point + lambda * center_out, flag);
    } else if ((sdf_in - thresh_vals[1]) * (sdf_out-thresh_vals[1]) <= 0.0f) {
        lambda = 0.5f * (abs(sdf_in - thresh_vals[1])/(abs(sdf_in - thresh_vals[1]) + abs(sdf_out - thresh_vals[1])));
        return vec4((1.0-lambda)*curr_point + lambda * center_out, flag);
    }

    return (length(center_out) < 1.4f) ? vec4((1.0-lambda)*curr_point + lambda * center_out, flag) : vec4(curr_point, flag);
}

void main() {
    // Use unsigned int value
    int myIndex = int(inIndex);
    vec4 curr_center = Center(myIndex);
    outPosition = curr_center.xyz;
    outAttributes = curr_center.w;
}
