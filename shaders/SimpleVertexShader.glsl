#version 330 core

layout(location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 uv;

uniform mat4 projection;
uniform mat4 view;

void main() {
    uv = aTexCoord;         // Map from [-1,1] to [0,1]
    gl_Position = projection * view * vec4(aPos, 1.0); //* view * model // vec4(aPos, 1.0); //
}