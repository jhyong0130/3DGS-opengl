#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 v_center[];
in float v_opacity[];
in vec3 v_dc[];
in mat2 v_cov[];

//uniform float GSize;

out vec2 f_uv;
out float f_opacity;
out vec3 f_dc;
out mat2 f_cov;

void main() {
    vec3 center = v_center[0];
    vec2 apos = center.xy;

    // Step 2: Compute scaling factor for 95% confidence
    float scale = sqrt(5.991f); // Chi-squared quantile

    if (center.z > 0.0f) {
        mat2 cov_in = v_cov[0];

        // 1. Compute eigenvalues
        float trace = cov_in[0][0] + cov_in[1][1];
        float det = cov_in[0][0] * cov_in[1][1] - cov_in[0][1] * cov_in[1][0];
        float delta = sqrt((trace * trace) / 4.0f - det);

        float lambda1 = trace / 2.0f + delta;
        float lambda2 = trace / 2.0f - delta;

        // 2. Compute eigenvectors

        // First eigenvector (for eigenvalue1)
        vec2 v1;
        if (abs(cov_in[0][1]) > 1e-6f) {
            v1 = normalize(vec2(lambda1 - cov_in[1][1], cov_in[0][1]));
        }
        else {
            v1 = vec2(1.0f, 0.0f);
        }

        // Second eigenvector (orthogonal to the first)
        vec2 v2 = normalize(vec2(-v1.y, v1.x )); // Perpendicular

        // Step 3: Get ellipse axes lengths
        float a = sqrt(lambda1) * scale;
        float b = sqrt(lambda2) * scale;

        // Step 4: Form ellipse axes vectors
        vec2 axis1 = v1 * a;
        vec2 axis2 = v2 * b;

        // Step 5: Get the 4 points
        vec2 p1 = axis1;
        vec2 p2 = - axis1;
        vec2 p3 = axis2;
        vec2 p4 = - axis2;

        // Step 6: Compute axis-aligned bounding box
        float minX = min(min(p1.x, p2.x), min(p3.x, p4.x));
        float maxX = max(max(p1.x, p2.x), max(p3.x, p4.x));
        float minY = min(min(p1.y, p2.y), min(p3.y, p4.y));
        float maxY = max(max(p1.y, p2.y), max(p3.y, p4.y));

        // Step 5: Get the 4 points
        gl_Position = vec4(apos + vec2(minX, minY), center.z, 1.0);

        f_uv = vec2(minX, minY);

        f_opacity = v_opacity[0];
        f_dc = v_dc[0];

        f_cov = v_cov[0];
        EmitVertex();

        // p2
        gl_Position = vec4(apos + vec2(maxX, minY), center.z, 1.0);

        f_uv = vec2(maxX, minY);

        f_opacity = v_opacity[0];
        f_dc = v_dc[0];

        f_cov = v_cov[0];
        EmitVertex();

        // p3
        gl_Position = vec4(apos + vec2(minX, maxY), center.z, 1.0);

        f_uv = vec2(minX, maxY);

        f_opacity = v_opacity[0];
        f_dc = v_dc[0];

        f_cov = v_cov[0];
        EmitVertex();

        // p4
        gl_Position = vec4(apos + vec2(maxX, maxY), center.z, 1.0);

        f_uv = vec2(maxX, maxY);

        f_opacity = v_opacity[0];
        f_dc = v_dc[0];

        f_cov = v_cov[0];
        EmitVertex();

        /*vec2 offsets[4] = vec2[](
            vec2(-1, -1), vec2(1, -1),
            vec2(-1, 1), vec2(1, 1)
        );

        vec2 apos;
        for (int i = 0; i < 4; ++i) {
            vec2 offset = offsets[i] * GSize;
            apos = center.xy + offset;

            gl_Position = vec4(apos.xy, center.z, 1.0);

            f_uv = offset;

            f_opacity = v_opacity[0];
            f_dc = v_dc[0];

            f_cov = v_cov[0];
            EmitVertex();
        }*/
    }
    EndPrimitive();
}