#version 330 core
out vec4 fragColor;
in vec2 uv;

uniform sampler2D accumTex;

void main() {

    vec4 accum = texture(accumTex, uv);
    float alpha = accum.a;
    vec3 color = accum.rgb; //(alpha > 0.0) ? accum.rgb / alpha : accum.rgb; //vec3(0.3); accum.rgb; //
    fragColor = vec4(color, 1.0); //vec4(color, 1.0);

    /*float size = 0.005;
    if (uv.x  <= size || uv.x >= 1.0-size ||
        uv.y  <= size || uv.y >= 1.0-size) {
        fragColor = vec4(vec3(1.0), 1.0);
    } else {
        vec2 offsets[8] = vec2[](
            vec2(-1, -1), vec2(1, -1),
            vec2(-1, 1), vec2(1, 1),
            vec2(-1, 0), vec2(1, 0),
            vec2(0, 1), vec2(0, 1)
        );

        bool local_max = true;
        vec2 apos;
        for (int i = 0; i < 8; ++i) {
            apos = uv + offsets[i]*size;

            accum = texture(accumTex, apos);
            alpha = accum.a;
            vec3 color_curr = accum.rgb; //(alpha > 0.0) ? accum.rgb / alpha : accum.rgb;
            if (color_curr.r > color.r + 1.0e-4) {
                local_max = false;
                break;
            }
        }

        fragColor = local_max ? vec4(vec3(0.0), 1.0) : vec4(vec3(1.0), 1.0);
    }*/
}