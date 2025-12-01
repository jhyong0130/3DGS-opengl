//
// Created by Briac on 28/08/2025.
//

#include <cuda_runtime.h>

#include "Backward.cuh"

#include "RenderingBase/CudaBuffer.cuh"
#include <cub/cub.cuh>

struct __align__(8) half4 {
    __half x, y, z, w;
};


//__device__ float dot(const float3& a, const float3& b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z;
//}

struct mat3;
__device__ mat3 transpose(const mat3& m);


struct mat3 {
    float3 rows[3];

    __device__ float3 operator*(const float3& v) const {
        return make_float3(
            dot(rows[0], v),
            dot(rows[1], v),
            dot(rows[2], v)
        );
    }

    __device__ mat3 operator*(const mat3& other) const {
        mat3 result;
        mat3 transB = transpose(other);
        for (int i = 0; i < 3; ++i) {
            result.rows[i] = make_float3(
                dot(rows[i], transB.rows[0]),
                dot(rows[i], transB.rows[1]),
                dot(rows[i], transB.rows[2])
            );
        }
        return result;
    }
};

__device__ mat3 transpose(const mat3& m) {
    mat3 t;
    t.rows[0] = make_float3(m.rows[0].x, m.rows[1].x, m.rows[2].x);
    t.rows[1] = make_float3(m.rows[0].y, m.rows[1].y, m.rows[2].y);
    t.rows[2] = make_float3(m.rows[0].z, m.rows[1].z, m.rows[2].z);
    return t;
}

__device__ mat3 computeCov3D(const float4& covX, const float4& covY, const float4& covZ, float mod, const mat3& viewMat) {
    mat3 R;
    R.rows[0] = make_float3(mod * covX.x, mod * covX.y, mod * covX.z);
    R.rows[1] = make_float3(mod * covY.x, mod * covY.y, mod * covY.z);
    R.rows[2] = make_float3(mod * covZ.x, mod * covZ.y, mod * covZ.z);

    mat3 viewT = transpose(viewMat);
    mat3 Sigma = viewMat * R * viewT;

    return Sigma;
}

struct mat2x3 {
    float2 rows[3];
};

__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, const mat3& cov3D) {
    float invZ = 1.0f / mean.z;
    float invZ2 = invZ * invZ;

    // Jacobian matrix J (2x3)
    float2 J_row0 = make_float2(focal_x * invZ, 0.0f);
    float2 J_row1 = make_float2(0.0f, focal_y * invZ);
    float2 J_row2 = make_float2(-focal_x * mean.x * invZ2, -focal_y * mean.y * invZ2);

    // Transpose(J) * cov3D * J
    float a = J_row0.x * dot(cov3D.rows[0], make_float3(J_row0.x, J_row1.x, J_row2.x)) +
        J_row0.y * dot(cov3D.rows[1], make_float3(J_row0.x, J_row1.x, J_row2.x));

    float b = J_row0.x * dot(cov3D.rows[0], make_float3(J_row0.y, J_row1.y, J_row2.y)) +
        J_row0.y * dot(cov3D.rows[1], make_float3(J_row0.y, J_row1.y, J_row2.y));

    float c = J_row1.x * dot(cov3D.rows[0], make_float3(J_row1.x, J_row1.y, J_row2.y)) +
        J_row1.y * dot(cov3D.rows[1], make_float3(J_row1.x, J_row1.y, J_row2.y));

    return make_float3(a, b, c);  // vec3(cov[0][0], cov[0][1], cov[1][1])
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
};
__device__ const float SH_C3[] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
};

// Sigmoid function applied to float3
__device__ float3 sigmoid(const float3& x) {
    return make_float3(
        1.0f / (1.0f + expf(-x.x)),
        1.0f / (1.0f + expf(-x.y)),
        1.0f / (1.0f + expf(-x.z))
    );
}

// Warp-level clustered sum of float3 across groups of 16 threads
__device__ float3 warp_cluster_sum_16(float3 val, int t_id) {
    // Lane ID within the warp
    int lane = t_id % 32;

    // All threads in warp participate
    unsigned mask = 0xFFFFFFFF;

    // Reduce in steps of 1, 2, 4, 8
    for (int offset = 1; offset < 16; offset *= 2) {
        float3 other;
        other.x = __shfl_down_sync(mask, val.x, offset, 16);
        other.y = __shfl_down_sync(mask, val.y, offset, 16);
        other.z = __shfl_down_sync(mask, val.z, offset, 16);

        // Only threads that remain in the 16-wide cluster add
        if ((lane % 16) + offset < 16) {
            val.x += other.x;
            val.y += other.y;
            val.z += other.z;
        }
    }

    // Now all threads in the 16-wide cluster have their partial sum
    return val;
}

__global__ void backprop_kernel(float4* position, float* sh_coeffs_red, float* sh_coeffs_green, float* sh_coeffs_blue, float4* covX, float4* covY, float4* covZ, float* SDF,
    half4* dLoss_dpredicted_colors, half4* dLoss_dconic_opacity, int* gaussian_indices, int* sorted_gaussian_indices,
    float* camera_pos, float* dLoss_sh_coeffs_R, float* dLoss_sh_coeffs_G, float* dLoss_sh_coeffs_B, float* dLoss_SDF,
    float* viewMat, float width, float height, float focal_x, float focal_y, float scale_modifier, float SDF_scale, int antialiasing, int nbPts) { // 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = idx / 16;
    const int k = idx % 16;

    if (n >= nbPts)
        return;

    int gauss_id = sorted_gaussian_indices[n]; //sorted_gaussian_indices[gaussian_indices[idx]];
    half4 dloss_color = dLoss_dpredicted_colors[n];
    half4 dloss_conic = dLoss_dconic_opacity[n];

    float4 P = position[gauss_id];

    const float4 dir = normalize(P - make_float4(camera_pos[0], camera_pos[1], camera_pos[2], camera_pos[3]));
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;

    const float3 sh_coeff = make_float3(
        sh_coeffs_red[gauss_id * 16 + k],
        sh_coeffs_green[gauss_id * 16 + k],
        sh_coeffs_blue[gauss_id * 16 + k]
    );

    float weight = 0.0f;
    if (k == 0) weight = SH_C0;
    if (k == 1) weight = -SH_C1 * y;
    if (k == 2) weight = SH_C1 * z;
    if (k == 3) weight = -SH_C1 * x;
    if (k == 4) weight = SH_C2[0] * xy;
    if (k == 5) weight = SH_C2[1] * yz;
    if (k == 6) weight = SH_C2[2] * (2.0f * zz - xx - yy);
    if (k == 7) weight = SH_C2[3] * xz;
    if (k == 8) weight = SH_C2[4] * (xx - yy);
    if (k == 9) weight = SH_C3[0] * y * (3.0f * xx - yy);
    if (k == 10) weight = SH_C3[1] * xy * z;
    if (k == 11) weight = SH_C3[2] * y * (4.0f * zz - xx - yy);
    if (k == 12) weight = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    if (k == 13) weight = SH_C3[4] * x * (4.0f * zz - xx - yy);
    if (k == 14) weight = SH_C3[5] * z * (xx - yy);
    if (k == 15) weight = SH_C3[6] * x * (xx - 3.0f * yy);

    // Clustered sum of sh_coeff * weight
    float3 val = make_float3(
        sh_coeff.x * weight,
        sh_coeff.y * weight,
        sh_coeff.z * weight
    );
    const float3 cluster_sum = warp_cluster_sum_16(val, (threadIdx.x & 31));

    // Apply sigmoid
    /*const float3 s = sigmoid(cluster_sum);

    dLoss_sh_coeffs_R[gauss_id * 16 + k] = weight * float(dloss_color.x) * s.x * (1.0f - s.x);
    dLoss_sh_coeffs_G[gauss_id * 16 + k] = weight * float(dloss_color.y) * s.y * (1.0f - s.y);
    dLoss_sh_coeffs_B[gauss_id * 16 + k] = weight * float(dloss_color.z) * s.z * (1.0f - s.z);*/

    dLoss_sh_coeffs_R[gauss_id * 16 + k] = weight * float(dloss_color.x);
    dLoss_sh_coeffs_G[gauss_id * 16 + k] = weight * float(dloss_color.y);
    dLoss_sh_coeffs_B[gauss_id * 16 + k] = weight * float(dloss_color.z);

    if (k == 0) {
        const float3 s = sigmoid(cluster_sum);
        for (int j = 0; j < 16; j++) {
            dLoss_sh_coeffs_R[gauss_id * 16 + j] = dLoss_sh_coeffs_R[gauss_id * 16 + j] * s.x * (1.0f - s.x);
            dLoss_sh_coeffs_G[gauss_id * 16 + j] = dLoss_sh_coeffs_G[gauss_id * 16 + j] * s.y * (1.0f - s.y);
            dLoss_sh_coeffs_B[gauss_id * 16 + j] = dLoss_sh_coeffs_B[gauss_id * 16 + j] * s.z * (1.0f - s.z);
        }


        /*if (0.5f + cluster_sum.x <= 0.0f) {
            for (int j = 0; j < 16; j++)
                dLoss_sh_coeffs_R[gauss_id * 16 + j] = 0.0f;
        }

        if (0.5f + cluster_sum.y <= 0.0f) {
            for (int j = 0; j < 16; j++)
                dLoss_sh_coeffs_G[gauss_id * 16 + j] = 0.0f;
        }

        if (0.5f + cluster_sum.z <= 0.0f) {
            for (int j = 0; j < 16; j++)
                dLoss_sh_coeffs_B[gauss_id * 16 + j] = 0.0f;
        }*/
    }

    if (k == 0) {
        // transform to view space
        mat3 viewMat3;
        viewMat3.rows[0] = make_float3(viewMat[0], viewMat[4], viewMat[8]);
        viewMat3.rows[1] = make_float3(viewMat[1], viewMat[5], viewMat[9]);
        viewMat3.rows[2] = make_float3(viewMat[2], viewMat[6], viewMat[10]);

        const float3 mean = make_float3(viewMat[0] * P.x + viewMat[4] * P.y + viewMat[8] * P.z + viewMat[12],
                                        viewMat[1] * P.x + viewMat[5] * P.y + viewMat[9] * P.z + viewMat[13], 
                                        viewMat[2] * P.x + viewMat[6] * P.y + viewMat[10] * P.z + viewMat[14]);

        const mat3 cov3D = computeCov3D(covX[gauss_id], covY[gauss_id], covZ[gauss_id], scale_modifier, viewMat3);

        // Compute 2D screen-space covariance matrix
        float3 cov = computeCov2D(mean, focal_x, focal_y, cov3D);

        const float h_var = 0.3f;
        const float det_cov = cov.x * cov.z - cov.y * cov.y;
        cov.x += h_var;
        cov.z += h_var;
        const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
        float h_convolution_scaling = 1.0f;

        if (antialiasing > 0)
            h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

        /*float sdf_val = SDF[gauss_id];
        float x = sdf_val / SDF_scale;
        float d_opacity = -2.0f * sdf_val / (SDF_scale * SDF_scale) * expf(-x * x);
        d_opacity *= (fabsf(sdf_val) <= SDF_scale) ? 1.0f : 0.0f;*/

        float sdf_val = SDF[gauss_id];
        float x = sdf_val * SDF_scale;
        float den = 1.0f + expf(-x);
        float d_opacity = -4.0f * SDF_scale * expf(-x) * (1.0f - expf(-x)) / (den*den*den);

        dLoss_SDF[gauss_id] = (h_convolution_scaling * float(dloss_conic.w)) * d_opacity;
    }
}

void Bwd::backprop(GLBuffer& position_b, GLBuffer& sh_coeffs_red, GLBuffer& sh_coeffs_green, GLBuffer& sh_coeffs_blue, GLBuffer& covX_b, GLBuffer& covY_b, GLBuffer& covZ_b, GLBuffer& SDF_b,
    GLBuffer& dLoss_dpredicted_colors, GLBuffer& dLoss_dconic_opacity, GLBuffer& gaussian_indices, GLBuffer& sorted_gaussian_indices,
    float* camera_pos, float* dLoss_sh_coeffs_R, float* dLoss_sh_coeffs_G, float* dLoss_sh_coeffs_B, float* dLoss_SDF,
    float* viewMat, float width, float height, float focal_x, float focal_y, float scale_modifier, float SDF_scale, int antialiasing, int count) {
    glFinish();

    checkCudaErrors(cudaGraphicsMapResources(1, &position_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &sh_coeffs_red.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &sh_coeffs_green.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &sh_coeffs_blue.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &covX_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &covY_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &covZ_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &SDF_b.getCudaResource()));

    checkCudaErrors(cudaGraphicsMapResources(1, &dLoss_dpredicted_colors.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &dLoss_dconic_opacity.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &gaussian_indices.getCudaResource()));
    checkCudaErrors(cudaGraphicsMapResources(1, &sorted_gaussian_indices.getCudaResource()));

    CudaBuffer<float4> position = CudaBuffer<float4>::fromGLBuffer(position_b);
    CudaBuffer<float> sh_coeffs_r = CudaBuffer<float>::fromGLBuffer(sh_coeffs_red);
    CudaBuffer<float> sh_coeffs_g = CudaBuffer<float>::fromGLBuffer(sh_coeffs_green);
    CudaBuffer<float> sh_coeffs_b = CudaBuffer<float>::fromGLBuffer(sh_coeffs_blue);
    CudaBuffer<float4> covX = CudaBuffer<float4>::fromGLBuffer(covX_b);
    CudaBuffer<float4> covY = CudaBuffer<float4>::fromGLBuffer(covY_b);
    CudaBuffer<float4> covZ = CudaBuffer<float4>::fromGLBuffer(covZ_b);
    CudaBuffer<float> SDF = CudaBuffer<float>::fromGLBuffer(SDF_b);

    CudaBuffer<half4> loss_color = CudaBuffer<half4>::fromGLBuffer(dLoss_dpredicted_colors);
    CudaBuffer<half4> loss_opacity = CudaBuffer<half4>::fromGLBuffer(dLoss_dconic_opacity);
    CudaBuffer<int> keys_in = CudaBuffer<int>::fromGLBuffer(gaussian_indices);
    CudaBuffer<int> keys_sorted = CudaBuffer<int>::fromGLBuffer(sorted_gaussian_indices);

    // Launch the kernel with
    int threadsPerBlock = 256;  // Safer default
    int blocksPerGrid = (16*count + threadsPerBlock - 1) / threadsPerBlock;
    backprop_kernel << < blocksPerGrid, threadsPerBlock >> > (position.ptr, sh_coeffs_r.ptr, sh_coeffs_g.ptr, sh_coeffs_b.ptr, covX.ptr, covY.ptr, covZ.ptr, SDF.ptr,
        loss_color.ptr, loss_opacity.ptr, keys_in.ptr, keys_sorted.ptr,
        camera_pos, dLoss_sh_coeffs_R, dLoss_sh_coeffs_G, dLoss_sh_coeffs_B, dLoss_SDF,
        viewMat, width, height, focal_x, focal_y, scale_modifier, SDF_scale, antialiasing, count);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &position_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &sh_coeffs_red.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &sh_coeffs_green.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &sh_coeffs_blue.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &covX_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &covY_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &covZ_b.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &SDF_b.getCudaResource()));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &dLoss_dpredicted_colors.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &dLoss_dconic_opacity.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &gaussian_indices.getCudaResource()));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &sorted_gaussian_indices.getCudaResource()));
}
