#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 v_center[];
//in mat3 v_cov[];

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 p_center;
uniform vec3 p_u;
uniform vec3 p_v;

uniform float GSize;

out vec3 f_uv;
//out mat3 f_cov;

void main() {
    float size = GSize; //0.5; // screen-space radius
    vec2 offsets[4] = vec2[](
        vec2(-1, -1), vec2(1, -1),
        vec2(-1, 1), vec2(1, 1)
    );

    vec4 apos;
    vec2 p_uv;
    for (int i = 0; i < 4; ++i) {
        vec2 offset = offsets[i] * size;
        apos = gl_in[0].gl_Position;
        apos.xy += offset; //quad on the cut plane
        gl_Position = apos; //
        p_uv = (apos.xy / apos.w); 
        apos = vec4(p_center + p_uv.x*p_u + p_uv.y*p_v, 1.0); // from cut plane equation to 3D position
        //gl_Position = apos; //projection * view * model * apos;
        f_uv = apos.xyz - v_center[0];
        //f_cov = v_cov[0];
        EmitVertex();
    }
    EndPrimitive();
}
