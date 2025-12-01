// Vertex shader (used with Transform Feedback)
#version 330 core
layout(location = 0) in uint inIndex;
out vec4 outPosition;
out float outAttributes;

uniform int K_val;
uniform samplerBuffer vertices;
uniform usamplerBuffer adjacents;

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
    float a = sqrt(point.x*point.x + point.y * point.y);
    float b = sqrt((a-R_max) * (a - R_max) + point.z * point.z);

    return vec3(((a-R_max)/b) * (point.x/a), ((a - R_max) / b) * (point.y / a), point.z/b);
}

vec3 SDF_Grad_func(vec3 point) {
    //return SDF_Grad_Sphere(point);
    return SDF_Grad_Torus(point, 0.6f, 0.4f);
}

// isocahedron
vec3 isoca[60] = vec3[60](
    vec3(0, 0.525731, 0.850651), vec3(0.850651, 0, 0.525731), vec3(0.525731, 0.850651, 0),
    vec3(0, 0.525731, 0.850651), vec3(-0.525731, 0.850651, 0), vec3(0.525731, 0.850651, 0),
    vec3(0, 0.525731, 0.850651), vec3(0.850651, 0, 0.525731), vec3(0, -0.525731, 0.850651),
    vec3(0, 0.525731, 0.850651), vec3(-0.850651, 0, 0.525731), vec3(0, -0.525731, 0.850651),
    vec3(0, 0.525731, 0.850651), vec3(-0.850651, 0, 0.525731), vec3(-0.525731, 0.850651, 0),
    vec3(0, -0.525731, 0.850651), vec3(-0.850651, 0, 0.525731), vec3(-0.525731, -0.850651, 0),
    vec3(0, -0.525731, 0.850651), vec3(-0.525731, -0.850651, 0), vec3(0.525731, -0.850651, 0),
    vec3(0, -0.525731, 0.850651), vec3(0.525731, -0.850651, 0), vec3(0.850651, 0, 0.525731),
    vec3(0.850651, 0, 0.525731), vec3(0.525731, -0.850651, 0), vec3(0.850651, 0, -0.525731),
    vec3(0.850651, 0, 0.525731), vec3(0.525731, 0.850651, 0), vec3(0.850651, 0, -0.525731),
    vec3(0.525731, 0.850651, 0), vec3(0.850651, 0, -0.525731), vec3(0, 0.525731, -0.850651),
    vec3(0.525731, 0.850651, 0), vec3(0, 0.525731, -0.850651), vec3(-0.525731, 0.850651, 0),
    vec3(0, 0.525731, -0.850651), vec3(0.850651, 0, -0.525731), vec3(0, -0.525731, -0.850651),
    vec3(0, 0.525731, -0.850651), vec3(0, -0.525731, -0.850651), vec3(-0.850651, 0, -0.525731),
    vec3(0, 0.525731, -0.850651), vec3(-0.850651, 0, -0.525731), vec3(-0.525731, 0.850651, 0),
    vec3(-0.525731, 0.850651, 0), vec3(-0.850651, 0, -0.525731), vec3(-0.850651, 0, 0.525731),
    vec3(-0.850651, 0, -0.525731), vec3(-0.850651, 0, 0.525731), vec3(-0.525731, -0.850651, 0),
    vec3(-0.850651, 0, -0.525731), vec3(-0.525731, -0.850651, 0), vec3(0, -0.525731, -0.850651),
    vec3(0, -0.525731, -0.850651), vec3(0.525731, -0.850651, 0), vec3(0.850651, 0, -0.525731),
    vec3(0, -0.525731, -0.850651), vec3(0.525731, -0.850651, 0), vec3(-0.525731, -0.850651, 0)
);


vec4 getIntersection(vec3 curr_point, vec3 ray, int myIndex, inout float flag_val) {
    float min_dist = 1.0e32;
    ivec3 curr_ids = ivec3(-1,-1,-1);
    for (int j = 0; j < K_val; j++)
    {
        int base = K_val * myIndex + j;
        int texelIndex = base / 4;
        int component = base % 4;
        uvec4 adjTexel = texelFetch(adjacents, texelIndex);
        uint neighborID;
        if (component == 0) neighborID = adjTexel.r;
        else if (component == 1) neighborID = adjTexel.g;
        else if (component == 2) neighborID = adjTexel.b;
        else neighborID = adjTexel.a;
        if (int(neighborID) == myIndex) continue;

        vec3 point = texelFetch(vertices, int(neighborID)).xyz; 

        // Compute bisector
        vec3 center = 0.5*(curr_point + point);
        vec3 dir = point - curr_point;
        if (length(dir) < 1e-6f) continue; 
        vec3 planeNormal = normalize(dir);
        
        //  Compute ray/bisector intersection
        float denom = dot(planeNormal, ray);
        if (denom < 1e-6f) continue; // Lines are parallel or coincident

        float t = dot(planeNormal, center - curr_point) / denom;
        
        if (t >= 0.0 && t < min_dist) {
            min_dist = t;
            curr_ids.x = int(neighborID); //j
            curr_ids.y = -1;
            curr_ids.z = -1;
        } else if (abs(t - min_dist) < 1.0e-5 && curr_ids.y == -1) {
            curr_ids.y = int(neighborID);
        } else if (abs(t - min_dist) < 1.0e-5) {
            curr_ids.z = int(neighborID);
        }
    }

    float sdf_0 = SDF_func(curr_point);
    vec3 point = curr_ids.x == -1 ? vec3(0.0f) : texelFetch(vertices, int(curr_ids.x)).xyz; 
    float sdf_1 = curr_ids.x == -1 ? 0.0 : SDF_func(point);
    point = curr_ids.y == -1 ? vec3(0.0f) : texelFetch(vertices, int(curr_ids.y)).xyz; 
    float sdf_2 = curr_ids.y == -1 ? 0.0 : SDF_func(point);
    point = curr_ids.z == -1 ? vec3(0.0f) : texelFetch(vertices, int(curr_ids.z)).xyz; 
    float sdf_3 = curr_ids.z == -1 ? 0.0 : SDF_func(point);

    for (int thresh_lvl = 0; thresh_lvl < 2; thresh_lvl++) {
        float d0 = sdf_0 - thresh_vals[thresh_lvl];
        float d1 = sdf_1 - thresh_vals[thresh_lvl];
        float d2 = sdf_2 - thresh_vals[thresh_lvl];
        float d3 = sdf_3 - thresh_vals[thresh_lvl];
        if ((curr_ids.x != -1 && (d0 * d1 <= 0.0f)) ||
            (curr_ids.y != -1 && (d0 * d2 <= 0.0f)) ||
            (curr_ids.z != -1 && (d0 * d3 <= 0.0f))) {
            if (flag_val == 0.0f) {
                if (thresh_lvl == 0)
                    flag_val = d0 >= 0.0f ? 1.0f: 0.0f;
                else
                    flag_val = d0 <= 0.0f ? 2.0f: 0.0f;
            }
        }
    }

    return vec4(min_dist, float(curr_ids.x), float(curr_ids.y), float(curr_ids.z));
}

bool test_indices(vec4 inter1, vec4 inter2, vec4 inter3) {
    bool inter1_2, inter1_3;
    
    if (inter1.z == -1.0f) {
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w );
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w );
    } else if (inter1.w == -1.0f){
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w ) ||
                    ( inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w );
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w ) ||
                    ( inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w );
    } else {
        inter1_2 = ( inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w ) ||
                    ( inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w ) ||
                    ( inter1.w == inter2.y || inter1.w == inter2.z || inter1.w == inter2.w ) ;
        inter1_3 = ( inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w ) ||
                        ( inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w ) ||
                        ( inter1.w == inter3.y || inter1.w == inter3.z || inter1.w == inter3.w ) ;
    }
    
    return inter1_2 && inter1_3;
}

vec3 Clip(vec3 p, vec3 in_vec, float l, vec3 ray, float thresh) {
    float norm = length(in_vec);
    if (norm < thresh) {
        for (int thresh_lvl = 0; thresh_lvl < 2; thresh_lvl++) {
            if ((SDF_func(p) - thresh_vals[thresh_lvl]) *
                (SDF_func(in_vec) - thresh_vals[thresh_lvl]) <= 0.0f) {
                vec3 grad_sdf = SDF_Grad_func(in_vec);
                float disp = dot(grad_sdf, ray);
                if (abs(disp) < 1e-6) return in_vec;

                return in_vec - ((SDF_func(in_vec) - thresh_vals[thresh_lvl])/disp) * ray;
            }
        }
        return in_vec;
    }

    float A = dot(ray, ray);
    float B = 2.0 * dot(p, ray);
    float C = dot(p, p) - thresh*thresh;

    float discriminant = B*B - 4.0*A*C;
    if (discriminant < 0.0) {
        // No real solution
        return in_vec;
    }

    float sqrtD = sqrt(discriminant);
    float l2_1 = (-B + sqrtD) / (2.0 * A);
    float l2_2 = (-B - sqrtD) / (2.0 * A);

    // Choose the l2 that is positive
    float l2 = (l2_1 > 0.0) ? l2_1 : l2_2;

    vec3 q2 = p + l2 * ray;
    
    return q2;
}

vec4 computeSplit_2(vec3 curr_point, vec3 s1_in, vec3 s2_in, vec3 s3_in, 
                    vec3 ray1, vec3 ray2, vec3 ray3, 
                    int myIndex, inout float flag_val) {
    vec3 ctd = vec3(0.0); 
    vec3 ctd_tmp, AB, AC, AD;
    float volume = 0.0f;
    float voltmp = 0.0f; 

    // split tetrahedron in middle
    vec3 new_s = (s1_in + s2_in + s3_in)/3.0f;
    vec3 new_central_ray = normalize(new_s - curr_point);
    
    new_s = (s1_in + s2_in)/2.0f;
    vec3 new_e1_ray = normalize(new_s - curr_point);
    
    new_s = (s1_in + s3_in)/2.0f;
    vec3 new_e2_ray = normalize(new_s - curr_point);
    
    new_s = (s2_in + s3_in)/2.0f;
    vec3 new_e3_ray = normalize(new_s - curr_point);

    // compute splitted volumes
    //////////////// First tetrahedra //////////////// vec4 c1 = computeCentroid(curr_point, ray1, new_e1_ray, new_central_ray, myIndex, flag_val);
    vec4 inter1 = getIntersection(curr_point, ray1, myIndex, flag_val);
    vec4 inter2 = getIntersection(curr_point, new_e1_ray, myIndex, flag_val);
    vec4 inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    vec3 s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    vec3 s2 = Clip(curr_point, curr_point + inter2.x*new_e1_ray, inter2.x, new_e1_ray, 1.5f);
    vec3 s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp; 
    
    
    //////////////// Second tetrahedra ////////////////  vec4 c2 = computeCentroid(curr_point, ray1, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray1, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e2_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e2_ray, inter2.x, new_e2_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp;
    
    
    //////////////// Third tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray2, new_e1_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray2, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e1_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray2, inter1.x, ray2, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e1_ray, inter2.x, new_e1_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
   
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp;
    
    
    //////////////// Fourth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray2, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray2, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e3_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray2, inter1.x, ray2, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e3_ray, inter2.x, new_e3_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp;
    
    
    //////////////// Fifth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray3, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray3, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e3_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray3, inter1.x, ray3, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e3_ray, inter2.x, new_e3_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp;
    
    
    //////////////// Sixth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray3, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray3, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e2_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray3, inter1.x, ray3, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e2_ray, inter2.x, new_e2_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    
    ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp*voltmp;
    volume = volume + voltmp;
    
    if (volume > 0.0f)
        return vec4(ctd/volume, volume);
    else
        return vec4((curr_point + s1_in + s2_in + s3_in)/4.0f, volume);
}

vec4 computeSplit_1(vec3 curr_point, vec3 s1_in, vec3 s2_in, vec3 s3_in, 
                    vec3 ray1, vec3 ray2, vec3 ray3, 
                    int myIndex, inout float flag_val) {
    vec3 ctd = vec3(0.0); 
    vec3 ctd_tmp, AB, AC, AD;
    float volume = 0.0f;
    float voltmp = 0.0f; 

    // split tetrahedron in middle
    vec3 new_s = (s1_in + s2_in + s3_in)/3.0f;
    vec3 new_central_ray = normalize(new_s - curr_point);
    
    new_s = (s1_in + s2_in)/2.0f;
    vec3 new_e1_ray = normalize(new_s - curr_point);
    
    new_s = (s1_in + s3_in)/2.0f;
    vec3 new_e2_ray = normalize(new_s - curr_point);
    
    new_s = (s2_in + s3_in)/2.0f;
    vec3 new_e3_ray = normalize(new_s - curr_point);

    // compute splitted volumes
    //////////////// First tetrahedra //////////////// vec4 c1 = computeCentroid(curr_point, ray1, new_e1_ray, new_central_ray, myIndex, flag_val);
    vec4 inter1 = getIntersection(curr_point, ray1, myIndex, flag_val);
    vec4 inter2 = getIntersection(curr_point, new_e1_ray, myIndex, flag_val);
    vec4 inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    vec3 s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    vec3 s2 = Clip(curr_point, curr_point + inter2.x*new_e1_ray, inter2.x, new_e1_ray, 1.5f);
    vec3 s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp; 
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray1, new_e1_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    //////////////// Second tetrahedra ////////////////  vec4 c2 = computeCentroid(curr_point, ray1, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray1, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e2_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e2_ray, inter2.x, new_e2_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp;
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray1, new_e2_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    
    //////////////// Third tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray2, new_e1_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray2, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e1_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray2, inter1.x, ray2, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e1_ray, inter2.x, new_e1_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp;
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray2, new_e1_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    //////////////// Fourth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray2, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray2, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e3_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray2, inter1.x, ray2, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e3_ray, inter2.x, new_e3_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp;
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray2, new_e3_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    //////////////// Fifth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray3, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray3, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e3_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray3, inter1.x, ray3, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e3_ray, inter2.x, new_e3_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp;
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray3, new_e3_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    //////////////// Sixth tetrahedra ////////////////  vec4 c3 = computeCentroid(curr_point, ray3, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(curr_point, ray3, myIndex, flag_val);
    inter2 = getIntersection(curr_point, new_e2_ray, myIndex, flag_val);
    inter3 = getIntersection(curr_point, new_central_ray, myIndex, flag_val);
    s1 = Clip(curr_point, curr_point + inter1.x*ray3, inter1.x, ray3, 1.5f);
    s2 = Clip(curr_point, curr_point + inter2.x*new_e2_ray, inter2.x, new_e2_ray, 1.5f);
    s3 = Clip(curr_point, curr_point + inter3.x*new_central_ray, inter3.x, new_central_ray, 1.5f);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3)/4.0f;
        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp*voltmp;
        volume = volume + voltmp;
    } else {
        vec4 ctd_tmp = computeSplit_2(curr_point, s1, s2, s3, 
                                    ray3, new_e2_ray, new_central_ray, 
                                    myIndex, flag_val);

        ctd = ctd + ctd_tmp.xyz*ctd_tmp.w;
        volume = volume + ctd_tmp.w; 
    }
    
    if (volume > 0.0f)
        return vec4(ctd/volume, volume);
    else
        return vec4((curr_point + s1_in + s2_in + s3_in)/4.0f, volume);
}

vec4 computeCentroid(vec3 curr_point, vec3 ray1, vec3 ray2, vec3 ray3, int myIndex, inout float flag_val) {
    vec4 inter1 = getIntersection(curr_point, ray1, myIndex, flag_val);
    vec4 inter2 = getIntersection(curr_point, ray2, myIndex, flag_val);
    vec4 inter3 = getIntersection(curr_point, ray3, myIndex, flag_val);

    if (inter1.x == 0.0f || inter2.x == 0.0f || inter3.x == 0.0f)
        return vec4(curr_point, 1.0f);

    vec3 s1 = Clip(curr_point, curr_point + inter1.x*ray1, inter1.x, ray1, 1.5f);
    vec3 s2 = Clip(curr_point, curr_point + inter2.x*ray2, inter2.x, ray2, 1.5f);
    vec3 s3 = Clip(curr_point, curr_point + inter3.x*ray3, inter3.x, ray3, 1.5f);

    vec3 ctd = vec3(0.0); 
    vec3 AB, AC, AD;
    float volume = 0;

    if (test_indices(inter1, inter2, inter3)) {
        ctd = (curr_point + s1 + s2 + s3)/4.0f;

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;

        volume = abs(dot(AB, cross(AC, AD)) / 6.0f);
        return vec4(ctd, volume);
    } else {
        return computeSplit_1(curr_point, s1, s2, s3, 
                                    ray1, ray2, ray3, 
                                    myIndex, flag_val);
    }
}

void main() {
    // Use unsigned int value
    int myIndex = int(inIndex);
    int id_v = myIndex / 20;
    int j = myIndex % 20;
    vec3 curr_point = texelFetch(vertices, id_v).xyz;
    float curr_flag = 0.0f;
    vec4 curr_center = computeCentroid(curr_point, isoca[3 * j], isoca[3 * j + 1], isoca[3 * j + 2], id_v, curr_flag);
    outPosition = curr_center;
    outAttributes = curr_flag;
}
