
#include <iostream>
#include "CVTupdate.cuh"
//#include <helper_math.h>

#include "RenderingBase/CudaBuffer.cuh"

__device__ float SDF_Sphere(float3 point, float R) {
    return length(point) - R;
}

__device__ float SDF_Torus(float3 point, float R_max, float R_min) {
    float qx = sqrt(point.x * point.x + point.y * point.y) - R_max;
    float qz = point.z;
    return sqrt(qx * qx + qz * qz) - R_min;
}

__device__ float SDF_Capsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Helper to rotate (y, z) vector
__device__ void rotate_yz(float* y, float* z, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    float newY = c * (*y) - s * (*z);
    float newZ = s * (*y) + c * (*z);
    *y = newY;
    *z = newZ;
}

__device__ float SDF_Bean(float3 p) {
    // Define bean spine
    float3 a = make_float3(-0.5, 0.0, 0.0);
    float3 b = make_float3(0.5, 0.0, 0.0);

    // Warp space for asymmetry (bend in Y-Z plane)
    float bend = 5.0f;
    float theta = bend * p.x; // bending angle
    rotate_yz(&p.y, &p.z, theta);

    return SDF_Capsule(p, a, b, 0.3f);
}

__device__ float SDF_func(float3 point) {
    //return SDF_Sphere(point, 0.5f);
    return SDF_Torus(point, 0.6f, 0.4f);
    //return SDF_Bean(point);
}

__device__ float3 SDF_Grad_Sphere(float3 point) {
    return normalize(point);
}

__device__ float3 SDF_Grad_Torus(float3 point, float R_max, float R_min) {
    float a = sqrt(point.x * point.x + point.y * point.y);
    float b = sqrt((a - R_max) * (a - R_max) + point.z * point.z);

    return make_float3(((a - R_max) / b) * (point.x / a), ((a - R_max) / b) * (point.y / a), point.z / b);
}

__device__ float3 SDF_Gradient_FD(float3 p) {
    const float eps = 1e-4f; // Small step
    return normalize(make_float3(
        SDF_func(make_float3(p.x + eps, p.y, p.z)) - SDF_func(make_float3(p.x - eps, p.y, p.z)),
        SDF_func(make_float3(p.x, p.y + eps, p.z)) - SDF_func(make_float3(p.x, p.y - eps, p.z)),
        SDF_func(make_float3(p.x, p.y, p.z + eps)) - SDF_func(make_float3(p.x, p.y, p.z - eps))
    ));
}

__device__ float3 SDF_Grad_func(float3 point) {
    //return SDF_Grad_Sphere(point);
    return SDF_Grad_Torus(point, 0.6f, 0.4f);
    //return SDF_Gradient_FD(point);
}

// isocahedron
__constant__ float3 isoca[60];
float3 h_isoca[60] = {
    make_float3(0, 0.525731, 0.850651), make_float3(0.850651, 0, 0.525731), make_float3(0.525731, 0.850651, 0),
    make_float3(0, 0.525731, 0.850651), make_float3(-0.525731, 0.850651, 0), make_float3(0.525731, 0.850651, 0),
    make_float3(0, 0.525731, 0.850651), make_float3(0.850651, 0, 0.525731), make_float3(0, -0.525731, 0.850651),
    make_float3(0, 0.525731, 0.850651), make_float3(-0.850651, 0, 0.525731), make_float3(0, -0.525731, 0.850651),
    make_float3(0, 0.525731, 0.850651), make_float3(-0.850651, 0, 0.525731), make_float3(-0.525731, 0.850651, 0),
    make_float3(0, -0.525731, 0.850651), make_float3(-0.850651, 0, 0.525731), make_float3(-0.525731, -0.850651, 0),
    make_float3(0, -0.525731, 0.850651), make_float3(-0.525731, -0.850651, 0), make_float3(0.525731, -0.850651, 0),
    make_float3(0, -0.525731, 0.850651), make_float3(0.525731, -0.850651, 0), make_float3(0.850651, 0, 0.525731),
    make_float3(0.850651, 0, 0.525731), make_float3(0.525731, -0.850651, 0), make_float3(0.850651, 0, -0.525731),
    make_float3(0.850651, 0, 0.525731), make_float3(0.525731, 0.850651, 0), make_float3(0.850651, 0, -0.525731),
    make_float3(0.525731, 0.850651, 0), make_float3(0.850651, 0, -0.525731), make_float3(0, 0.525731, -0.850651),
    make_float3(0.525731, 0.850651, 0), make_float3(0, 0.525731, -0.850651), make_float3(-0.525731, 0.850651, 0),
    make_float3(0, 0.525731, -0.850651), make_float3(0.850651, 0, -0.525731), make_float3(0, -0.525731, -0.850651),
    make_float3(0, 0.525731, -0.850651), make_float3(0, -0.525731, -0.850651), make_float3(-0.850651, 0, -0.525731),
    make_float3(0, 0.525731, -0.850651), make_float3(-0.850651, 0, -0.525731), make_float3(-0.525731, 0.850651, 0),
    make_float3(-0.525731, 0.850651, 0), make_float3(-0.850651, 0, -0.525731), make_float3(-0.850651, 0, 0.525731),
    make_float3(-0.850651, 0, -0.525731), make_float3(-0.850651, 0, 0.525731), make_float3(-0.525731, -0.850651, 0),
    make_float3(-0.850651, 0, -0.525731), make_float3(-0.525731, -0.850651, 0), make_float3(0, -0.525731, -0.850651),
    make_float3(0, -0.525731, -0.850651), make_float3(0.525731, -0.850651, 0), make_float3(0.850651, 0, -0.525731),
    make_float3(0, -0.525731, -0.850651), make_float3(0.525731, -0.850651, 0), make_float3(-0.525731, -0.850651, 0)
};

__device__ float getMinSDF1D(float3 curr_point, float3 s1, float3 s2, float SDF1, float SDF2) {
    float3 u = s2 - curr_point;
    float3 v = s1 - s2;
    float delta_S = SDF1 - SDF2;
    float alpha = dot(u, v);
    float beta = v.x * v.x + v.y * v.y + v.z * v.z;
    float gamma = u.x * u.x + u.y * u.y + u.z * u.z;

    float A = beta * beta - delta_S * delta_S * beta;
    float B = 2.0f * alpha * (beta - delta_S * delta_S);
    float C = alpha * alpha - delta_S * delta_S * gamma;

    if (A == 0.0f) {
        if (B == 0.0f) {
            // degenerate case
            return 1.0e32f;
        }
        else {
            float min_eval = 1.0e32f;
            float lambda = C / B;
            if (lambda >= 0 && lambda <= 1.0f &&
                fabs(dot(v, u + lambda * v) + delta_S * length(u + lambda * v)) < 1.0e-6f) {
                float curr_f = SDF2 + lambda * delta_S + length(u + lambda * v);
                if (curr_f < min_eval) {
                    min_eval = curr_f;
                }
            }
            return min_eval;
        }
        return 1.0e32f;
    }

    float min_eval = 1.0e32f;
    if (B * B - 4.0f * A * C >= 0.0f) {
        float lambda0 = (-B - sqrt(B * B - 4.0f * A * C)) * (2.0f * A);
        float lambda1 = (-B + sqrt(B * B - 4.0f * A * C)) * (2.0f * A);

        if (lambda0 >= 0 && lambda0 <= 1.0f &&
            fabs(dot(v, u + lambda0 * v) + delta_S * length(u + lambda0 * v)) < 1.0e-6f) {
            float curr_f = SDF2 + lambda0 * delta_S + length(u + lambda0 * v);
            if (curr_f < min_eval) {
                min_eval = curr_f;
            }
        }

        if (lambda1 >= 0 && lambda1 <= 1.0f &&
            fabs(dot(v, u + lambda1 * v) + delta_S * length(u + lambda1 * v)) < 1.0e-6f) {
            float curr_f = SDF2 + lambda1 * delta_S + length(u + lambda1 * v);
            if (curr_f < min_eval) {
                min_eval = curr_f;
            }
        }
    }
    else {
        if (SDF1 < min_eval) {
            min_eval = SDF1;
        }
        if (SDF2 < min_eval) {
            min_eval = SDF2;
        }
    }

    return min_eval;
}


__device__ float getMinSDF(float3 curr_point, float3 s1, float3 s2, float3 s3, 
                            float SDF1, float SDF2, float SDF3) {
    float delta_a = SDF1 - SDF3;
    float delta_b = SDF2 - SDF3;

    float3 v1 = s1 - s3;
    float3 v2 = s2 - s3;
    float3 w = s3 - curr_point;

    float2 A_0 = make_float2(dot(v1, v1), dot(v1,v2));
    float2 A_1 = make_float2(dot(v2, v1), dot(v2, v2));
    float2 b = make_float2(dot(v1, w), dot(v2, w));

    float det_A = A_0.x * A_1.y - A_1.x * A_0.y;
    if (det_A == 0.0f) {
        // Treat here degenerate case
        /*If A is singular (collinear vertices), reduce to a segment problem (or point) and treat accordingly.
            If a=0 (i.e.q!=1), the r-equation becomes linear; handle that as the linear special case.
            Numerically, solve the quadratic robustly and test feasibility; the problem is convex (linear SDF + convex norm), so this search finds the global minimizer.
            Practically: implement the algebra above once and fall back to a small numeric solver (Newton) if you prefer simpler code — but the closed-form quadratic + linear solves give an analytic candidate set.*/
        return 1.0e32f;
    }

    float2 Ainv_0 = make_float2(A_1.y/det_A, -A_1.x / det_A);
    float2 Ainv_1 = make_float2(-A_0.y / det_A, A_0.x / det_A);

    float2 VAinv_0 = make_float2(v1.x* Ainv_0.x + v1.y* Ainv_1.x, v1.x * Ainv_0.y + v1.y * Ainv_1.y);
    float2 VAinv_1 = make_float2(v2.x * Ainv_0.x + v2.y * Ainv_1.x, v2.x * Ainv_0.y + v2.y * Ainv_1.y);

    float2 q = make_float2(VAinv_0.x*delta_a + VAinv_0.y*delta_b, VAinv_1.x * delta_a + VAinv_1.y * delta_b);
    float2 p0 = make_float2(-(VAinv_0.x * b.x + VAinv_0.y * b.y) + w.x, -(VAinv_1.x * b.x + VAinv_1.y * b.y) + w.y);

    float alpha = 1.0f - (q.x*q.x + q.y*q.y);
    float beta = 2.0f * dot(q, p0);
    float gamma = -(p0.x * p0.x + p0.y * p0.y);

    float min_eval = 1.0e32f;
    if (beta * beta - 4.0f * alpha * gamma >= 0.0f) {
        float r0 = (-beta + sqrt(beta * beta - 4.0f * alpha * gamma)) / (2.0f * alpha);
        float r1 = (-beta - sqrt(beta * beta - 4.0f * alpha * gamma)) / (2.0f * alpha);

        if (r0 > 0.0f) {
            float u = -r0 * (VAinv_0.x * delta_a + VAinv_0.y * delta_b) - VAinv_0.x * b.x + VAinv_0.y * b.y;
            float v = -r0 * (VAinv_1.x * delta_a + VAinv_1.y * delta_b) - VAinv_1.x * b.x + VAinv_1.y * b.y;
            float3 p = u * v1 + v * v2 + w;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f &&
                fabs(dot(v1, p) + delta_a * r0) < 1.0e-6f && fabs(dot(v1, p) + delta_a * r0) < 1.0e-6f) {
                float f_curr = u * SDF1 + v * SDF2 + (1.0f - u - v) * SDF3 + r0;
                if (f_curr < min_eval) {
                    min_eval = f_curr;
                }
            }
        }

        if (r1 > 0.0f) {
            float u = -r1 * (VAinv_0.x * delta_a + VAinv_0.y * delta_b) - VAinv_0.x * b.x + VAinv_0.y * b.y;
            float v = -r1 * (VAinv_1.x * delta_a + VAinv_1.y * delta_b) - VAinv_1.x * b.x + VAinv_1.y * b.y;
            float3 p = u * v1 + v * v2 + w;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f &&
                fabs(dot(v1, p) + delta_a * r1) < 1.0e-6f && fabs(dot(v1, p) + delta_a * r1) < 1.0e-6f) {
                float f_curr = u * SDF1 + v * SDF2 + (1.0f - u - v) * SDF3 + r1;
                if (f_curr < min_eval) {
                    min_eval = f_curr;
                }
            }
        }
    }

    if (min_eval == 1.0e32f) {
        /*If no feasible interior solution is found, the minimizer lies on the triangle boundary. Handle by:
            Minimize on each edge (segment) using the 1-D solution you already have for segments ab, bc, and ca (set one barycentric coordinate to 0 and solve the 1D problem).
            Also evaluate the three vertices a,b,c.
            The global minimizer is the smallest among interior candidate (if any), edge minima, and vertex values.*/

        float f_curr = getMinSDF1D(curr_point, s1, s2, SDF1, SDF2);
        if (f_curr < min_eval) {
            min_eval = f_curr;
        }

        f_curr = getMinSDF1D(curr_point, s1, s3, SDF1, SDF3);
        if (f_curr < min_eval) {
            min_eval = f_curr;
        }

        f_curr = getMinSDF1D(curr_point, s2, s3, SDF2, SDF3);
        if (f_curr < min_eval) {
            min_eval = f_curr;
        }

        if (SDF1 < min_eval) {
            min_eval = SDF1;
        }
        if (SDF2 < min_eval) {
            min_eval = SDF2;
        }
        if (SDF3 < min_eval) {
            min_eval = SDF3;
        }

        return min_eval;
    }

    return min_eval;
}

__device__ float4 getIntersection(float4* vertices, uint4* adjacents, float3 curr_point, float3 ray, int myIndex, unsigned char* flag_val, int K_val, float *thresh_vals) {
    float min_dist = 1.0e32;
    int3 curr_ids = make_int3(-1, -1, -1);
    for (int j = 0; j < K_val; j++)
    {
        int base = K_val * myIndex + j;
        int texelIndex = base / 4;
        int component = base % 4;
        uint4 adjTexel = adjacents[texelIndex];
        uint neighborID;
        if (component == 0) neighborID = adjTexel.x;
        else if (component == 1) neighborID = adjTexel.y;
        else if (component == 2) neighborID = adjTexel.z;
        else neighborID = adjTexel.w;
        if (int(neighborID) == myIndex) continue;

        float3 point = make_float3(vertices[int(neighborID)].x, vertices[int(neighborID)].y, vertices[int(neighborID)].z);

        // Compute bisector
        float3 center = 0.5 * (curr_point + point);
        float3 dir = point - curr_point;
        if (length(dir) < 1e-6f) continue;
        float3 planeNormal = normalize(dir);

        //  Compute ray/bisector intersection
        float denom = dot(planeNormal, ray);
        if (denom < 1e-6f) continue; // Lines are parallel or coincident

        float t = dot(planeNormal, center - curr_point) / denom;

        if (t >= 0.0 && t < min_dist) {
            min_dist = t;
            curr_ids.x = int(neighborID); //j
            curr_ids.y = -1;
            curr_ids.z = -1;
        }
        else if (abs(t - min_dist) < 1.0e-5 && curr_ids.y == -1) {
            curr_ids.y = int(neighborID);
        }
        else if (abs(t - min_dist) < 1.0e-5) {
            curr_ids.z = int(neighborID);
        }
    }

    float sdf_0 = SDF_func(curr_point);
    float3 point = curr_ids.x == -1 ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(vertices[int(curr_ids.x)].x, vertices[int(curr_ids.x)].y, vertices[int(curr_ids.x)].z);
    float sdf_1 = curr_ids.x == -1 ? 0.0 : SDF_func(point);
    point = curr_ids.y == -1 ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(vertices[int(curr_ids.y)].x, vertices[int(curr_ids.y)].y, vertices[int(curr_ids.y)].z);
    float sdf_2 = curr_ids.y == -1 ? 0.0 : SDF_func(point);
    point = curr_ids.z == -1 ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(vertices[int(curr_ids.z)].x, vertices[int(curr_ids.z)].y, vertices[int(curr_ids.z)].z);
    float sdf_3 = curr_ids.z == -1 ? 0.0 : SDF_func(point);

    for (int thresh_lvl = 0; thresh_lvl < 2; thresh_lvl++) {
        float d0 = sdf_0 - thresh_vals[thresh_lvl];
        float d1 = sdf_1 - thresh_vals[thresh_lvl];
        float d2 = sdf_2 - thresh_vals[thresh_lvl];
        float d3 = sdf_3 - thresh_vals[thresh_lvl];
        if ((curr_ids.x != -1 && (d0 * d1 <= 0.0f)) ||
            (curr_ids.y != -1 && (d0 * d2 <= 0.0f)) ||
            (curr_ids.z != -1 && (d0 * d3 <= 0.0f))) {
            if (*flag_val == 0) {
                if (thresh_lvl == 0)
                    *flag_val = d0 >= 0.0f ? 1 : 0;
                else
                    *flag_val = d0 <= 0.0f ? 2 : 0;
            } else if (*flag_val == 1) {
                if (thresh_lvl == 0)
                    *flag_val = d0 >= 0.0f ? 1 : 1;
                else
                    *flag_val = d0 <= 0.0f ? 3 : 1;
            } else if (*flag_val == 2) {
                if (thresh_lvl == 0)
                    *flag_val = d0 >= 0.0f ? 3 : 2;
                else
                    *flag_val = d0 <= 0.0f ? 2 : 2;
            }
        }
    }

    return make_float4(min_dist, float(curr_ids.x), float(curr_ids.y), float(curr_ids.z));
}

__device__ bool test_indices(float4 inter1, float4 inter2, float4 inter3) {
    bool inter1_2, inter1_3;

    if (inter1.z == -1.0f) {
        inter1_2 = (inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w);
        inter1_3 = (inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w);
    }
    else if (inter1.w == -1.0f) {
        inter1_2 = (inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w) ||
            (inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w);
        inter1_3 = (inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w) ||
            (inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w);
    }
    else {
        inter1_2 = (inter1.y == inter2.y || inter1.y == inter2.z || inter1.y == inter2.w) ||
            (inter1.z == inter2.y || inter1.z == inter2.z || inter1.z == inter2.w) ||
            (inter1.w == inter2.y || inter1.w == inter2.z || inter1.w == inter2.w);
        inter1_3 = (inter1.y == inter3.y || inter1.y == inter3.z || inter1.y == inter3.w) ||
            (inter1.z == inter3.y || inter1.z == inter3.z || inter1.z == inter3.w) ||
            (inter1.w == inter3.y || inter1.w == inter3.z || inter1.w == inter3.w);
    }

    return inter1_2 && inter1_3;
}

__device__ float3 Clip(float3 p, float3 in_vec, float l, float3 ray, float thresh, float* thresh_vals) {
    float norm = length(in_vec);
    if (norm < thresh) {
        for (int thresh_lvl = 0; thresh_lvl < 2; thresh_lvl++) {
            if ((SDF_func(p) - thresh_vals[thresh_lvl]) *
                (SDF_func(in_vec) - thresh_vals[thresh_lvl]) <= 0.0f) {
                float3 grad_sdf = SDF_Grad_func(in_vec);
                float disp = dot(grad_sdf, ray);
                if (abs(disp) < 1e-6) return in_vec;

                return in_vec - ((SDF_func(in_vec) - thresh_vals[thresh_lvl]) / disp) * ray;
            }
        }
        return in_vec;
    }

    float A = dot(ray, ray);
    float B = 2.0 * dot(p, ray);
    float C = dot(p, p) - thresh * thresh;

    float discriminant = B * B - 4.0 * A * C;
    if (discriminant < 0.0) {
        // No real solution
        return in_vec;
    }

    float sqrtD = sqrt(discriminant);
    float l2_1 = (-B + sqrtD) / (2.0 * A);
    float l2_2 = (-B - sqrtD) / (2.0 * A);

    // Choose the l2 that is positive
    float l2 = (l2_1 > 0.0) ? l2_1 : l2_2;

    float3 q2 = p + l2 * ray;

    return q2;
}

__device__ float4 computeSplit_2(float4* vertices, uint4* adjacents, float3 curr_point, float3 s1_in, float3 s2_in, float3 s3_in,
    float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in, float3 ray1, float3 ray2, float3 ray3,
    int myIndex, float* min_sdf, unsigned char* flag_val, int K_val, float* thresh_vals) {
    float3 ctd = make_float3(0.0f, 0.0f, 0.0f);
    float3 ctd_tmp, AB, AC, AD;
    float volume = 0.0f;
    float voltmp = 0.0f;

    // split tetrahedron in middle
    float3 new_s = (s1_in + s2_in + s3_in) / 3.0f;
    float3 new_central_ray = normalize(new_s - curr_point);

    new_s = (s1_in + s2_in) / 2.0f;
    float3 new_e1_ray = normalize(new_s - curr_point);

    new_s = (s1_in + s3_in) / 2.0f;
    float3 new_e2_ray = normalize(new_s - curr_point);

    new_s = (s2_in + s3_in) / 2.0f;
    float3 new_e3_ray = normalize(new_s - curr_point);

    // compute splitted volumes
    //////////////// First tetrahedra //////////////// float4 c1 = computeCentroid(curr_point, ray1, new_e1_ray, new_central_ray, myIndex, flag_val);
    float4 inter1 = getIntersection(vertices, adjacents, curr_point, ray1, myIndex, flag_val, K_val, thresh_vals);
    float4 inter2 = getIntersection(vertices, adjacents, curr_point, new_e1_ray, myIndex, flag_val, K_val, thresh_vals);
    float4 inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    float3 s1 = Clip(curr_point, curr_point + inter1.x * ray1, inter1.x, ray1, 1.5f, thresh_vals);
    float3 s2 = Clip(curr_point, curr_point + inter2.x * new_e1_ray, inter2.x, new_e1_ray, 1.5f, thresh_vals);
    float3 s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    float3 mid;

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;


    //////////////// Second tetrahedra ////////////////  float4 c2 = computeCentroid(curr_point, ray1, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray1, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e2_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray1, inter1.x, ray1, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e2_ray, inter2.x, new_e2_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;


    //////////////// Third tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray2, new_e1_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray2, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e1_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray2, inter1.x, ray2, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e1_ray, inter2.x, new_e1_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;


    //////////////// Fourth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray2, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray2, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e3_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray2, inter1.x, ray2, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e3_ray, inter2.x, new_e3_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;


    //////////////// Fifth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray3, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray3, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e3_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray3, inter1.x, ray3, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e3_ray, inter2.x, new_e3_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;


    //////////////// Sixth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray3, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray3, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e2_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray3, inter1.x, ray3, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e2_ray, inter2.x, new_e2_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    // compute min SDF
    min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

    ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
    // compute volume of tetrahedra
    AB = s1 - curr_point;
    AC = s2 - curr_point;
    AD = s3 - curr_point;
    voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

    ctd = ctd + ctd_tmp * voltmp;
    volume = volume + voltmp;

    // Compute covariance matrix
    mid = (curr_point + s1 + s2 + s3) / 4.0f;
    covMatrixX_in[0].x += voltmp * mid.x * mid.x;
    covMatrixX_in[0].y += voltmp * mid.x * mid.y;
    covMatrixX_in[0].z += voltmp * mid.x * mid.z;

    covMatrixY_in[0].x += voltmp * mid.y * mid.x;
    covMatrixY_in[0].y += voltmp * mid.y * mid.y;
    covMatrixY_in[0].z += voltmp * mid.y * mid.z;

    covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
    covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
    covMatrixZ_in[0].z += voltmp * mid.z * mid.z;

    if (volume > 0.0f)
        return make_float4(ctd / volume, volume);
    else
        return make_float4((curr_point + s1_in + s2_in + s3_in) / 4.0f, volume);
}

__device__ float4 computeSplit_1(float4* vertices, uint4* adjacents, float3 curr_point, float3 s1_in, float3 s2_in, float3 s3_in,
    float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in, float3 ray1, float3 ray2, float3 ray3,
    int myIndex, float* min_sdf, unsigned char* flag_val, int K_val, float* thresh_vals) {
    float3 ctd = make_float3(0.0f, 0.0f, 0.0f);
    float3 ctd_tmp, AB, AC, AD;
    float volume = 0.0f;
    float voltmp = 0.0f;

    // split tetrahedron in middle
    float3 new_s = (s1_in + s2_in + s3_in) / 3.0f;
    float3 new_central_ray = normalize(new_s - curr_point);

    new_s = (s1_in + s2_in) / 2.0f;
    float3 new_e1_ray = normalize(new_s - curr_point);

    new_s = (s1_in + s3_in) / 2.0f;
    float3 new_e2_ray = normalize(new_s - curr_point);

    new_s = (s2_in + s3_in) / 2.0f;
    float3 new_e3_ray = normalize(new_s - curr_point);

    // compute splitted volumes
    //////////////// First tetrahedra //////////////// float4 c1 = computeCentroid(curr_point, ray1, new_e1_ray, new_central_ray, myIndex, flag_val);
    float4 inter1 = getIntersection(vertices, adjacents, curr_point, ray1, myIndex, flag_val, K_val, thresh_vals);
    float4 inter2 = getIntersection(vertices, adjacents, curr_point, new_e1_ray, myIndex, flag_val, K_val, thresh_vals);
    float4 inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    float3 s1 = Clip(curr_point, curr_point + inter1.x * ray1, inter1.x, ray1, 1.5f, thresh_vals);
    float3 s2 = Clip(curr_point, curr_point + inter2.x * new_e1_ray, inter2.x, new_e1_ray, 1.5f, thresh_vals);
    float3 s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);

    float3 mid;

    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray1, new_e1_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }

    //////////////// Second tetrahedra ////////////////  float4 c2 = computeCentroid(curr_point, ray1, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray1, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e2_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray1, inter1.x, ray1, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e2_ray, inter2.x, new_e2_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray1, new_e2_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }


    //////////////// Third tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray2, new_e1_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray2, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e1_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray2, inter1.x, ray2, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e1_ray, inter2.x, new_e1_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray2, new_e1_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }

    //////////////// Fourth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray2, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray2, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e3_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray2, inter1.x, ray2, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e3_ray, inter2.x, new_e3_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray2, new_e3_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }

    //////////////// Fifth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray3, new_e3_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray3, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e3_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray3, inter1.x, ray3, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e3_ray, inter2.x, new_e3_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray3, new_e3_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }

    //////////////// Sixth tetrahedra ////////////////  float4 c3 = computeCentroid(curr_point, ray3, new_e2_ray, new_central_ray, myIndex, flag_val);
    inter1 = getIntersection(vertices, adjacents, curr_point, ray3, myIndex, flag_val, K_val, thresh_vals);
    inter2 = getIntersection(vertices, adjacents, curr_point, new_e2_ray, myIndex, flag_val, K_val, thresh_vals);
    inter3 = getIntersection(vertices, adjacents, curr_point, new_central_ray, myIndex, flag_val, K_val, thresh_vals);
    s1 = Clip(curr_point, curr_point + inter1.x * ray3, inter1.x, ray3, 1.5f, thresh_vals);
    s2 = Clip(curr_point, curr_point + inter2.x * new_e2_ray, inter2.x, new_e2_ray, 1.5f, thresh_vals);
    s3 = Clip(curr_point, curr_point + inter3.x * new_central_ray, inter3.x, new_central_ray, 1.5f, thresh_vals);
    if (test_indices(inter1, inter2, inter3)) {
        ctd_tmp = (curr_point + s1 + s2 + s3) / 4.0f;
        // compute min SDF
        min_sdf[0] = min(min_sdf[0], getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3)));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;
        voltmp = abs(dot(AB, cross(AC, AD)) / 6.0f);

        ctd = ctd + ctd_tmp * voltmp;
        volume = volume + voltmp;

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += voltmp * mid.x * mid.x;
        covMatrixX_in[0].y += voltmp * mid.x * mid.y;
        covMatrixX_in[0].z += voltmp * mid.x * mid.z;

        covMatrixY_in[0].x += voltmp * mid.y * mid.x;
        covMatrixY_in[0].y += voltmp * mid.y * mid.y;
        covMatrixY_in[0].z += voltmp * mid.y * mid.z;

        covMatrixZ_in[0].x += voltmp * mid.z * mid.x;
        covMatrixZ_in[0].y += voltmp * mid.z * mid.y;
        covMatrixZ_in[0].z += voltmp * mid.z * mid.z;
    }
    else {
        float4 ctd_tmp = computeSplit_2(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray3, new_e2_ray, new_central_ray,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);

        ctd = ctd + make_float3(ctd_tmp.x, ctd_tmp.y, ctd_tmp.z) * ctd_tmp.w;
        volume = volume + ctd_tmp.w;
    }

    if (volume > 0.0f)
        return make_float4(ctd / volume, volume);
    else
        return make_float4((curr_point + s1_in + s2_in + s3_in) / 4.0f, volume);
}

__device__ float4 computeCentroid(float4* vertices, uint4* adjacents, float3 curr_point, float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in,
    float3 ray1, float3 ray2, float3 ray3, int myIndex, float* min_sdf, unsigned char* flag_val, int K_val, float* thresh_vals) {
    float4 inter1 = getIntersection(vertices, adjacents, curr_point, ray1, myIndex, flag_val, K_val, thresh_vals);
    float4 inter2 = getIntersection(vertices, adjacents, curr_point, ray2, myIndex, flag_val, K_val, thresh_vals);
    float4 inter3 = getIntersection(vertices, adjacents, curr_point, ray3, myIndex, flag_val, K_val, thresh_vals);

    if (inter1.x == 0.0f || inter2.x == 0.0f || inter3.x == 0.0f)
        return make_float4(curr_point, 1.0f);

    float3 s1 = Clip(curr_point, curr_point + inter1.x * ray1, inter1.x, ray1, 1.5f, thresh_vals);
    float3 s2 = Clip(curr_point, curr_point + inter2.x * ray2, inter2.x, ray2, 1.5f, thresh_vals);
    float3 s3 = Clip(curr_point, curr_point + inter3.x * ray3, inter3.x, ray3, 1.5f, thresh_vals);

    float3 ctd = make_float3(0.0f, 0.0f, 0.0f);
    float3 AB, AC, AD, mid;
    float volume = 0.0f;

    if (test_indices(inter1, inter2, inter3)) {
        ctd = (curr_point + s1 + s2 + s3) / 4.0f;

        // compute min SDF
        min_sdf[0] = getMinSDF(curr_point, s1, s2, s3, SDF_func(s1), SDF_func(s2), SDF_func(s3));

        // compute volume of tetrahedra
        AB = s1 - curr_point;
        AC = s2 - curr_point;
        AD = s3 - curr_point;

        volume = abs(dot(AB, cross(AC, AD)) / 6.0f);

        // Compute covariance matrix
        mid = (curr_point + s1 + s2 + s3) / 4.0f;
        covMatrixX_in[0].x += volume * mid.x * mid.x;
        covMatrixX_in[0].y += volume * mid.x * mid.y;
        covMatrixX_in[0].z += volume * mid.x * mid.z;

        covMatrixY_in[0].x += volume * mid.y * mid.x;
        covMatrixY_in[0].y += volume * mid.y * mid.y;
        covMatrixY_in[0].z += volume * mid.y * mid.z;

        covMatrixZ_in[0].x += volume * mid.z * mid.x;
        covMatrixZ_in[0].y += volume * mid.z * mid.y;
        covMatrixZ_in[0].z += volume * mid.z * mid.z;

        return make_float4(ctd, volume);
    }
    else {
        return computeSplit_1(vertices, adjacents, curr_point, s1, s2, s3,
            covMatrixX_in, covMatrixY_in, covMatrixZ_in, ray1, ray2, ray3,
            myIndex, min_sdf, flag_val, K_val, thresh_vals);
    }
}

__global__ void update_kernel(float* sdf, float4* vertices, uint4* adjacents, float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in,
    unsigned char* flags, int K_val, float* thresh_vals, int nbPts) { // 
    // One thread per ray
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx == 0)
    //    printf("Hello from GPU thread %d!\n",  threadIdx.x);
    //    printf("Hello from GPU thread %d, %f, %f, %f!\n", idx, vertices[idx].x, vertices[idx].y, vertices[idx].z); //threadIdx.x);


    int id_v = idx / 20;
    int id_j = idx % 20;

    if (id_v >= nbPts)
        return;


    float3 curr_point = make_float3(vertices[id_v].x, vertices[id_v].y, vertices[id_v].z);

    __shared__ float4 curr_centers[200];
    __shared__ float4 covMatrixX[200];
    __shared__ float4 covMatrixY[200];
    __shared__ float4 covMatrixZ[200];
    __shared__ unsigned char curr_flags[200];
    __shared__ float min_SDF[200];

    covMatrixX[threadIdx.x] = make_float4(0.0f);
    covMatrixY[threadIdx.x] = make_float4(0.0f);
    covMatrixZ[threadIdx.x] = make_float4(0.0f);
    curr_flags[threadIdx.x] = 0;
    min_SDF[threadIdx.x] = 1.0e32f;
    curr_centers[threadIdx.x] = computeCentroid(vertices, adjacents, curr_point, &covMatrixX[threadIdx.x], &covMatrixY[threadIdx.x], &covMatrixZ[threadIdx.x],
        isoca[3 * id_j], isoca[3 * id_j + 1], isoca[3 * id_j + 2], id_v, &min_SDF[threadIdx.x], &curr_flags[threadIdx.x], K_val, thresh_vals);

    __syncthreads();  // All threads must reach this point

    if (id_j == 0) {
        // Initialize covariance matrix to zero
        covMatrixX_in[id_v] = make_float4(0.0f);
        covMatrixY_in[id_v] = make_float4(0.0f);
        covMatrixZ_in[id_v] = make_float4(0.0f);

        float tot_area = 0.0f;
        float4 curr_center;
        float3 center_out = make_float3(0.0f, 0.0f, 0.0f);
        float min_sdf = 1.0e32f;

        unsigned char flag = 0;
        // loop through all 20 tetrehedra that tesselate the sphere
        for (int j = 0; j < 20; j++) {
            curr_center = curr_centers[20 * (threadIdx.x / 20) + j];
            if (flag == 0) {
                flag = curr_flags[20 * (threadIdx.x / 20) + j];
            }
            else if (flag == 1) {
                flag = curr_flags[20 * (threadIdx.x / 20) + j] == 2 ? 3 : 1;
            }
            else if (flag == 2) {
                flag = curr_flags[20 * (threadIdx.x / 20) + j] == 1 ? 3 : 2;
            }

            covMatrixX_in[id_v] = covMatrixX_in[id_v] + covMatrixX[20 * (threadIdx.x / 20) + j];
            covMatrixY_in[id_v] = covMatrixY_in[id_v] + covMatrixY[20 * (threadIdx.x / 20) + j];
            covMatrixZ_in[id_v] = covMatrixZ_in[id_v] + covMatrixZ[20 * (threadIdx.x / 20) + j];

            center_out = center_out + make_float3(curr_center.x, curr_center.y, curr_center.z) * curr_center.w;
            tot_area = tot_area + curr_center.w;

            min_sdf = min(min_sdf, min_SDF[20 * (threadIdx.x / 20) + j]);
        }

        if (tot_area > 0.0f) {
            center_out = center_out / tot_area;
        }


        // Compute Covariance matrix
        /*for (int j = 0; j < 20; j++) {
            curr_center = curr_centers[20 * (threadIdx.x / 20) + j];
            float3 diff = 1.0f*(make_float3(curr_center.x, curr_center.y, curr_center.z) - center_out);
            float w = curr_center.w;

            covMatrixX_in[id_v].x = covMatrixX_in[id_v].x + (w * diff.x * diff.x);
            covMatrixX_in[id_v].y = covMatrixX_in[id_v].y + (w * diff.x * diff.y);
            covMatrixX_in[id_v].z = covMatrixX_in[id_v].z + (w * diff.x * diff.z);

            covMatrixY_in[id_v].x = covMatrixY_in[id_v].x + (w * diff.y * diff.x);
            covMatrixY_in[id_v].y = covMatrixY_in[id_v].y + (w * diff.y * diff.y);
            covMatrixY_in[id_v].z = covMatrixY_in[id_v].z + (w * diff.y * diff.z);

            covMatrixZ_in[id_v].x = covMatrixZ_in[id_v].x + (w * diff.z * diff.x);
            covMatrixZ_in[id_v].y = covMatrixZ_in[id_v].y + (w * diff.z * diff.y);
            covMatrixZ_in[id_v].z = covMatrixZ_in[id_v].z + (w * diff.z * diff.z);
        }*/

        // Divide by total weight
        if (tot_area > 0.0f) {
            covMatrixX_in[id_v] = covMatrixX_in[id_v] / tot_area;
            covMatrixY_in[id_v] = covMatrixY_in[id_v] / tot_area;
            covMatrixZ_in[id_v] = covMatrixZ_in[id_v] / tot_area;
        }

        // Compute covariance matrix
        covMatrixX_in[id_v].x -= center_out.x * center_out.x;
        covMatrixX_in[id_v].y -= center_out.x * center_out.y;
        covMatrixX_in[id_v].z -= center_out.x * center_out.z;

        covMatrixY_in[id_v].x -= center_out.y * center_out.x;
        covMatrixY_in[id_v].y -= center_out.y * center_out.y;
        covMatrixY_in[id_v].z -= center_out.y * center_out.z;

        covMatrixZ_in[id_v].x -= center_out.z * center_out.x;
        covMatrixZ_in[id_v].y -= center_out.z * center_out.y;
        covMatrixZ_in[id_v].z -= center_out.z * center_out.z;


        float sdf_in = SDF_func(curr_point);
        float sdf_out = SDF_func(center_out);
        float lambda = 0.5f;

        flags[id_v] = flag;
        if ((sdf_in - thresh_vals[0]) * (sdf_out - thresh_vals[0]) <= 0.0f) {
            lambda = 0.5f * (abs(sdf_in - thresh_vals[0]) / (abs(sdf_in - thresh_vals[0]) + abs(sdf_out - thresh_vals[0])));
            vertices[id_v] = make_float4((1.0 - lambda) * curr_point + lambda * center_out, 1.0f);
            sdf[id_v] = ((1.0 - lambda) * curr_point + lambda * center_out).z - 1.0f;//SDF_func((1.0 - lambda) * curr_point + lambda * center_out);
            return;
        }
        else if ((sdf_in - thresh_vals[1]) * (sdf_out - thresh_vals[1]) <= 0.0f) {
            lambda = 0.5f * (abs(sdf_in - thresh_vals[1]) / (abs(sdf_in - thresh_vals[1]) + abs(sdf_out - thresh_vals[1])));
            vertices[id_v] = make_float4((1.0 - lambda) * curr_point + lambda * center_out, 1.0f);
            sdf[id_v] = ((1.0 - lambda) * curr_point + lambda * center_out).z - 1.0f;//SDF_func((1.0 - lambda) * curr_point + lambda * center_out);
            return;
        }

        if (length(center_out) < 1.4f) {
            vertices[id_v] = make_float4((1.0 - lambda) * curr_point + lambda * center_out, 1.0f);
        }
        sdf[id_v] = vertices[id_v].z - 1.0f;//SDF_func(make_float3(vertices[id_v].x, vertices[id_v].y, vertices[id_v].z));

        return;
    }

}

void CVT::update(GLBuffer& sdf, GLBuffer& vertices, uint4* adjacents, GLBuffer& covMatrixX, GLBuffer& covMatrixY, GLBuffer& covMatrixZ, unsigned char* flag, int K_val, float* thresh_vals, int count) {
    cudaMemcpyToSymbol(isoca, h_isoca, sizeof(float3) * 60);

    checkCudaErrors(cudaGraphicsMapResources(1, &sdf.getCudaResource()));

    CudaBuffer<float> sdf_in = CudaBuffer<float>::fromGLBuffer(sdf);
    CudaBuffer<float4> vertices_in = CudaBuffer<float4>::fromGLBuffer(vertices);
    CudaBuffer<float4> covMatrixX_in = CudaBuffer<float4>::fromGLBuffer(covMatrixX);
    CudaBuffer<float4> covMatrixY_in = CudaBuffer<float4>::fromGLBuffer(covMatrixY);
    CudaBuffer<float4> covMatrixZ_in = CudaBuffer<float4>::fromGLBuffer(covMatrixZ);

    // Launch the kernel with 1 block of 5 threads /20 + 1
    update_kernel << <count /10 + 1, 200 >> > (sdf_in.ptr, vertices_in.ptr, adjacents, covMatrixX_in.ptr, covMatrixY_in.ptr, covMatrixZ_in.ptr, flag, K_val, thresh_vals, count);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &sdf.getCudaResource()));
}


__global__ void copy_float4_float3(float3* out, float4* pts, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float4 p = pts[idx];
    out[idx].x = p.x;
    out[idx].y = p.y;
    out[idx].z = p.z;
}

void CVT::cpy_pts(GLBuffer& positions, float3* pts_f3, int count) {
    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    checkCudaErrors(cudaGraphicsMapResources(1, &positions.getCudaResource()));
    CudaBuffer<float4> vertices_in = CudaBuffer<float4>::fromGLBuffer(positions);
    copy_float4_float3 << < blocksPerGrid, threadsPerBlock >> > (pts_f3, vertices_in.ptr, count);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaGraphicsUnmapResources(1, &positions.getCudaResource()));

}

__global__ void minlvl_kernel(float* sdf, float* threshold_sdf, float* min_lvl, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float curr_sdf = sdf[idx];
    float diff0 = fabsf(curr_sdf - threshold_sdf[0]);
    float diff1 = fabsf(curr_sdf - threshold_sdf[1]);

    if ((curr_sdf - threshold_sdf[0]) > 0.0f) {
        atomicMinFloat(&min_lvl[0], diff0);
    }

    if ((curr_sdf - threshold_sdf[1]) > 0.0f) {
        atomicMinFloat(&min_lvl[1], diff1);
    }
}

void CVT::min_lvls(GLBuffer& sdf, float* threshold_sdf, float* min_lvl, int count) {
    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    checkCudaErrors(cudaGraphicsMapResources(1, &sdf.getCudaResource()));

    CudaBuffer<float> sdf_in = CudaBuffer<float>::fromGLBuffer(sdf);

    // Launch the kernel with 1 block of 5 threads /20 + 1
    minlvl_kernel << <blocksPerGrid, threadsPerBlock >> > (sdf_in.ptr, threshold_sdf, min_lvl, count);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &sdf.getCudaResource()));
}

void CVT::map_to_cpu(GLBuffer& vertices, float4* data, int count) {

    checkCudaErrors(cudaGraphicsMapResources(1, &vertices.getCudaResource()));

    CudaBuffer<float4> vertices_in = CudaBuffer<float4>::fromGLBuffer(vertices);

    cudaMemcpy(data, vertices_in.ptr, 4 * count * sizeof(float), cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &vertices.getCudaResource()));
}

void CVT::map_to_CUDA(GLBuffer& sdf, GLBuffer& vertices, GLBuffer& covMatrixX, GLBuffer& covMatrixY, GLBuffer& covMatrixZ,
                        GLBuffer& sh_coeffsR, GLBuffer& sh_coeffsG, GLBuffer& sh_coeffsB,
                        float* sdf_in, float4* vertices_in,
                        float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in,
                        float* sh_coeffsR_in, float* sh_coeffsG_in, float* sh_coeffsB_in, int num_gaussians) {

    checkCudaErrors(cudaGraphicsMapResources(1, &sdf.getCudaResource()));

    CudaBuffer<float> sdf_b = CudaBuffer<float>::fromGLBuffer(sdf);
    CudaBuffer<float4> vertices_b = CudaBuffer<float4>::fromGLBuffer(vertices);
    CudaBuffer<float4> covMatrixX_b = CudaBuffer<float4>::fromGLBuffer(covMatrixX);
    CudaBuffer<float4> covMatrixY_b = CudaBuffer<float4>::fromGLBuffer(covMatrixY);
    CudaBuffer<float4> covMatrixZ_b = CudaBuffer<float4>::fromGLBuffer(covMatrixZ);
    CudaBuffer<float> sh_coeffsR_b = CudaBuffer<float>::fromGLBuffer(sh_coeffsR);
    CudaBuffer<float> sh_coeffsG_b = CudaBuffer<float>::fromGLBuffer(sh_coeffsG);
    CudaBuffer<float> sh_coeffsB_b = CudaBuffer<float>::fromGLBuffer(sh_coeffsB);

    cudaMemcpy(sh_coeffsR_in, sh_coeffsR_b.ptr, 16 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sh_coeffsG_in, sh_coeffsG_b.ptr, 16 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sh_coeffsB_in, sh_coeffsB_b.ptr, 16 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(covMatrixX_in, covMatrixX_b.ptr, 4 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(covMatrixY_in, covMatrixY_b.ptr, 4 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(covMatrixZ_in, covMatrixZ_b.ptr, 4 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(sdf_in, sdf_b.ptr, num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vertices_in, vertices_b.ptr, 4 * num_gaussians * sizeof(float), cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &sdf.getCudaResource()));
}