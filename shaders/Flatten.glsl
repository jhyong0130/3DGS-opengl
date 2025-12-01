// Vertex shader (used with Transform Feedback)
#version 330 core

layout(location = 0) in vec3 in_position;

out vec3 outPosition;

uniform samplerBuffer vertices;

void main() {
    outPosition = in_position;
}
