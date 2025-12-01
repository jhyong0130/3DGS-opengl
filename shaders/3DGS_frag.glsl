#version 330 core
in vec2 f_uv;
in float f_opacity;
in vec3 f_dc;
//in vec3 f_color;
in mat2 f_cov;

out vec4 fragColor;

uniform float GScale;

void main() {
    // Elliptical Gaussian (screen-space)
    /*float Scale = GScale / 640.0f;
    mat2 covtmp;
    covtmp[0].xy = vec2(1.0f/(Scale*Scale), 0.0);
    covtmp[1].xy = vec2(0.0, 1.0f/(Scale*Scale));*/
    float alpha = f_opacity * //(1.0 / (1.0 + exp(-f_opacity))) *
                    exp(-0.5 * dot(f_uv, f_cov * f_uv)); // f_cov
    vec3 color = f_dc; //1.0 / (1.0 + exp(-f_dc)); // Sigmoid

    fragColor = vec4(color * alpha,  alpha);
}