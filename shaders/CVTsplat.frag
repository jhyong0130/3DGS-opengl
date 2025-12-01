#version 330 core
in vec3 f_uv;
in vec3 f_color;
//in mat3 f_cov;

out vec4 fragColor;

uniform float Scale;

void main() {
    vec3 uv = f_uv;

    // Elliptical Gaussian (screen-space)
    float alpha = 1.0;

    /*mat3 covtmp;
    covtmp[0].xyz = vec3(GSize, 0.0, 0.0);
    covtmp[1].xyz = vec3(0.0, GSize, 0.0);
    covtmp[2].xyz = vec3(0.0, 0.0, GSize);
    float exponent = -0.5 * dot(uv, covtmp * uv); // f_cov
    float k = GSize;*/
    float dist = min(1.0, sqrt(dot(uv, uv)));
    float density = 1.0f; //exp(-k*dist*dist);

    fragColor = vec4(Scale * vec3(dist*dist, dist*dist, dist*dist) * density * alpha, density * alpha);
}
