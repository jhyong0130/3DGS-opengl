#include "RgbdLoadCuda.cuh"
#include <cstdio>
#include <cmath>

// ============================================================
// CUDA kernel: unproject depth pixels to 3D, project to RGB,
// filter invalid, write compacted outputs using atomicAdd.
// ============================================================
__device__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float d_choose_scale_neus(float depth_mm, float3 n_cam, float3 p_cam_mm)
{
    const float D = 1500.0f;
    const float S_near = 1000000.0f;
    const float S_far = 100000.0f;
    const float alpha = 1.0f;

    float x = fminf(fmaxf(depth_mm / D, 0.0f), 1.0f);
    float s_dist = S_near - (S_near - S_far) * (x * x);

    float p_len = sqrtf(p_cam_mm.x * p_cam_mm.x + p_cam_mm.y * p_cam_mm.y + p_cam_mm.z * p_cam_mm.z);
    float n_len = sqrtf(n_cam.x * n_cam.x + n_cam.y * n_cam.y + n_cam.z * n_cam.z);
    float cos_nv = 0.05f;
    if (p_len > 1e-8f && n_len > 1e-8f) {
        float dot_nv = fabsf(-(n_cam.x * p_cam_mm.x + n_cam.y * p_cam_mm.y + n_cam.z * p_cam_mm.z) / (n_len * p_len));
        cos_nv = fmaxf(dot_nv, 0.05f);
    }

    float orient = fminf(fmaxf(1.0f + alpha / cos_nv, 1.0f), 6.0f);
    float s = s_dist * orient;
    return fminf(fmaxf(s, S_far), S_near);
}

__global__ void rgbdUnprojectKernel(
    RgbdLoadParams params,
    float* positions,
    float* normals,
    float* colors,
    float* opacities,
    float* tangents,
    float* depth_cam_points,
    float* scale_neus,
    int* valid_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = params.depth_width * params.depth_height;
    if (idx >= total) return;

    int i = idx / params.depth_width; // row (v)
    int j = idx % params.depth_width; // col (u)

    // Read depth
    float depth_mm = (float)params.depth_image[i * params.depth_width + j];
    if (depth_mm <= 0.0f || depth_mm > params.max_depth_mm) return;

    // Unproject to depth camera coords (mm)
    float x_d = (j - params.cx_depth) * depth_mm / params.fx_depth;
    float y_d = (i - params.cy_depth) * depth_mm / params.fy_depth;
    float z_d = depth_mm;

    // Transform to RGB camera coords
    float x_rgb = params.R[0]*x_d + params.R[1]*y_d + params.R[2]*z_d + params.T[0];
    float y_rgb = params.R[3]*x_d + params.R[4]*y_d + params.R[5]*z_d + params.T[1];
    float z_rgb = params.R[6]*x_d + params.R[7]*y_d + params.R[8]*z_d + params.T[2];

    if (z_rgb <= 0.0f) return;

    // Project to RGB image coords
    float u = (x_rgb * params.fx_rgb) / z_rgb + params.cx_rgb;
    float v = (y_rgb * params.fy_rgb) / z_rgb + params.cy_rgb;

    // Border check
    if (u < (float)params.border_left || u >= (float)(params.rgb_width - params.border_right)) return;
    if (v < (float)params.border_top  || v >= (float)(params.rgb_height - params.border_bottom)) return;

    int u_int = max(0, min(params.rgb_width  - 1, (int)roundf(u)));
    int v_int = max(0, min(params.rgb_height - 1, (int)roundf(v)));

    // ---- Compute normal from depth ----
    // Get neighbor depths
    float dz    = depth_mm;
    float dz_du = (j + 1 < params.depth_width)  ? (float)params.depth_image[i * params.depth_width + (j+1)] : -1.0f;
    float dz_dv = (i + 1 < params.depth_height)  ? (float)params.depth_image[(i+1) * params.depth_width + j] : -1.0f;

    if (dz_du <= 0.0f || dz_du > 10000.0f) return;
    if (dz_dv <= 0.0f || dz_dv > 10000.0f) return;

    // Points in depth camera coords
    float3 p  = make_float3((j - params.cx_depth) * dz / params.fx_depth,
                             (i - params.cy_depth) * dz / params.fy_depth, dz);
    float3 px = make_float3((j+1 - params.cx_depth) * dz_du / params.fx_depth,
                             (i - params.cy_depth) * dz_du / params.fy_depth, dz_du);
    float3 py = make_float3((j - params.cx_depth) * dz_dv / params.fx_depth,
                             (i+1 - params.cy_depth) * dz_dv / params.fy_depth, dz_dv);

    float3 vx = make_float3(px.x - p.x, px.y - p.y, px.z - p.z);
    float3 vy = make_float3(py.x - p.x, py.y - p.y, py.z - p.z);

    // Cross product vx ?~ vy
    float3 n;
    n.x = vx.y * vy.z - vx.z * vy.y;
    n.y = vx.z * vy.x - vx.x * vy.z;
    n.z = vx.x * vy.y - vx.y * vy.x;
    float n_len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (n_len < 1e-8f) return;
    n.x /= n_len; n.y /= n_len; n.z /= n_len;

    // Ensure normal faces the camera (positive Z)
    float3 t = vx;
    if (n.z < 0.0f) {
        n.x = -n.x; n.y = -n.y; n.z = -n.z;
        t.x = -t.x; t.y = -t.y; t.z = -t.z;
    }

    if (isnan(n.x) || isnan(n.y) || isnan(n.z)) return;

    // Convert normal to world coords: n_rgb = R_d2r * n_cam, then n_world = R_w * n_rgb
    float nx_rgb = params.R[0]*n.x + params.R[1]*n.y + params.R[2]*n.z;
    float ny_rgb = params.R[3]*n.x + params.R[4]*n.y + params.R[5]*n.z;
    float nz_rgb = params.R[6]*n.x + params.R[7]*n.y + params.R[8]*n.z;

    float nx_w = params.R_world[0]*nx_rgb + params.R_world[1]*ny_rgb + params.R_world[2]*nz_rgb;
    float ny_w = params.R_world[3]*nx_rgb + params.R_world[4]*ny_rgb + params.R_world[5]*nz_rgb;
    float nz_w = params.R_world[6]*nx_rgb + params.R_world[7]*ny_rgb + params.R_world[8]*nz_rgb;
    float nw_len = sqrtf(nx_w*nx_w + ny_w*ny_w + nz_w*nz_w);
    if (nw_len < 1e-8f) return;
    nx_w /= nw_len; ny_w /= nw_len; nz_w /= nw_len;

    // Transform point_rgb to world
    float wx = params.R_world[0]*x_rgb + params.R_world[1]*y_rgb + params.R_world[2]*z_rgb + params.T_world[0];
    float wy = params.R_world[3]*x_rgb + params.R_world[4]*y_rgb + params.R_world[5]*z_rgb + params.T_world[1];
    float wz = params.R_world[6]*x_rgb + params.R_world[7]*y_rgb + params.R_world[8]*z_rgb + params.T_world[2];

    // Opacity (1.0 until 1000mm, then decay)
    float opacity;
    if (depth_mm <= 1000.0f) {
        opacity = 1.0f;
    } else {
        float lambda = -logf(0.01f) / (2.0f - 1.0f);
        opacity = expf(-lambda * (depth_mm - 1000.0f));
    }

    // Get color (BGR format)
    int pixel_idx = (v_int * params.rgb_width + u_int) * params.rgb_channels;
    float b_col = (float)params.rgb_image[pixel_idx + 0] / 255.0f;
    float g_col = (float)params.rgb_image[pixel_idx + 1] / 255.0f;
    float r_col = (float)params.rgb_image[pixel_idx + 2] / 255.0f;

    // Atomically get output index
    int out_idx = atomicAdd(valid_count, 1);

    // Write position in OpenGL convention (flip y, z)
    positions[out_idx * 4 + 0] = wx;
    positions[out_idx * 4 + 1] = -wy;  // OpenGL flip
    positions[out_idx * 4 + 2] = -wz;  // OpenGL flip
    positions[out_idx * 4 + 3] = 1.0f;

    // Write normal in OpenGL convention
    normals[out_idx * 4 + 0] = nx_w;
    normals[out_idx * 4 + 1] = -ny_w;
    normals[out_idx * 4 + 2] = -nz_w;
    normals[out_idx * 4 + 3] = 0.0f;

    // Write color (RGB order)
    colors[out_idx * 3 + 0] = r_col;
    colors[out_idx * 3 + 1] = g_col;
    colors[out_idx * 3 + 2] = b_col;

    // Write opacity
    opacities[out_idx] = opacity;

    // Write tangent in camera space
    tangents[out_idx * 3 + 0] = t.x;
    tangents[out_idx * 3 + 1] = t.y;
    tangents[out_idx * 3 + 2] = t.z;

    // Write depth camera point
    depth_cam_points[out_idx * 3 + 0] = x_d;
    depth_cam_points[out_idx * 3 + 1] = y_d;
    depth_cam_points[out_idx * 3 + 2] = z_d;

    // NeuS sharpness parameter (matches CPU choose_scale_neus)
    scale_neus[out_idx] = d_choose_scale_neus(depth_mm, n, p);
}

int launchRgbdUnprojectKernel(
    const RgbdLoadParams& params,
    RgbdLoadOutputs& outputs,
    cudaStream_t stream)
{
    int total_pixels = params.depth_width * params.depth_height;
    int blockSize = 256;
    int gridSize = (total_pixels + blockSize - 1) / blockSize;

    // Zero the counter
    cudaMemsetAsync(outputs.valid_count, 0, sizeof(int), stream);

    rgbdUnprojectKernel<<<gridSize, blockSize, 0, stream>>>(
        params,
        outputs.positions,
        outputs.normals,
        outputs.colors,
        outputs.opacities,
        outputs.tangents,
        outputs.depth_cam_points,
        outputs.scale_neus,
        outputs.valid_count
    );

    cudaStreamSynchronize(stream);

    // Read back count
    int count = 0;
    cudaMemcpy(&count, outputs.valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

// ============================================================
// CUDA kernel: compute depth-dependent covariance per point,
// then transform to world coords.
// ============================================================
__global__ void depthCovarianceKernel(
    const float* d_depth_cam_points,
    const float* d_tangents,
    float fx_depth, float fy_depth,
    const float* R_d2r,   // 9 floats row-major
    const float* R_world,  // 9 floats row-major
    float* d_covX, float* d_covY, float* d_covZ,
    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const float depth_noise_a = 0.0007f;
    const float depth_noise_b = 0.2f;
    const float scale_factor = 1.0f;

    float z_mm = fabsf(d_depth_cam_points[idx * 3 + 2]);

    float s_x = z_mm / fx_depth;
    float s_y = z_mm / fy_depth;

    float tx = d_tangents[idx * 3 + 0];
    float ty = d_tangents[idx * 3 + 1];
    float s_z = (s_x * tx + s_y * ty) / 2.0f;
    if (s_z <= 0.05f || isnan(s_z) || isinf(s_z))
        s_z = depth_noise_a * z_mm - depth_noise_b;

    float var_x = (s_x * s_x) / 4.0f;
    float var_y = (s_y * s_y) / 4.0f;
    float var_z = (s_z * s_z) / 4.0f;

    // Diagonal covariance in depth camera space
    // C_depth = diag(var_x, var_y, var_z) * scale_factor
    float C_d[9] = {
        var_x * scale_factor, 0.0f, 0.0f,
        0.0f, var_y * scale_factor, 0.0f,
        0.0f, 0.0f, var_z * scale_factor
    };

    // C_rgb = R * C_depth * R^T
    float RC[9]; // R * C_depth
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += R_d2r[r*3+k] * C_d[k*3+c];
            }
            RC[r*3+c] = sum;
        }
    }
    float C_rgb[9]; // RC * R^T
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += RC[r*3+k] * R_d2r[c*3+k]; // R^T column = R row
            }
            C_rgb[r*3+c] = sum;
        }
    }

    // C_world = R_world * C_rgb * R_world^T
    float RC2[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += R_world[r*3+k] * C_rgb[k*3+c];
            }
            RC2[r*3+c] = sum;
        }
    }
    float C_world[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += RC2[r*3+k] * R_world[c*3+k];
            }
            C_world[r*3+c] = sum;
        }
    }

    // Write covariance rows
    d_covX[idx*4+0] = C_world[0]; d_covX[idx*4+1] = C_world[1]; d_covX[idx*4+2] = C_world[2]; d_covX[idx*4+3] = 0.0f;
    d_covY[idx*4+0] = C_world[3]; d_covY[idx*4+1] = C_world[4]; d_covY[idx*4+2] = C_world[5]; d_covY[idx*4+3] = 0.0f;
    d_covZ[idx*4+0] = C_world[6]; d_covZ[idx*4+1] = C_world[7]; d_covZ[idx*4+2] = C_world[8]; d_covZ[idx*4+3] = 0.0f;
}

void launchDepthCovarianceKernel(
    const float* d_depth_cam_points,
    const float* d_tangents,
    float fx_depth, float fy_depth,
    const float* R_d2r,
    const float* T_d2r,
    const float* R_world,
    float* d_covX, float* d_covY, float* d_covZ,
    int num_points,
    cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (num_points + blockSize - 1) / blockSize;

    // Upload rotation matrices to device
    float* d_R_d2r;
    float* d_R_world;
    cudaMalloc(&d_R_d2r, 9 * sizeof(float));
    cudaMalloc(&d_R_world, 9 * sizeof(float));
    cudaMemcpy(d_R_d2r, R_d2r, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_world, R_world, 9 * sizeof(float), cudaMemcpyHostToDevice);

    depthCovarianceKernel<<<gridSize, blockSize, 0, stream>>>(
        d_depth_cam_points, d_tangents,
        fx_depth, fy_depth,
        d_R_d2r, d_R_world,
        d_covX, d_covY, d_covZ,
        num_points
    );

    cudaStreamSynchronize(stream);

    cudaFree(d_R_d2r);
    cudaFree(d_R_world);
}
