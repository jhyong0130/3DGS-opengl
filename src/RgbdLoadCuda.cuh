#ifndef RGBD_LOAD_CUDA_CUH
#define RGBD_LOAD_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Parameters for the RGBD unprojection kernel
struct RgbdLoadParams {
    // Depth image (device pointer, uint16)
    const uint16_t* depth_image;
    int depth_width;
    int depth_height;

    // RGB image (device pointer, RGBA8 or BGR8 with 3 channels)
    const unsigned char* rgb_image;
    int rgb_width;
    int rgb_height;
    int rgb_channels; // 3 for BGR

    // Depth camera intrinsics
    float fx_depth, fy_depth, cx_depth, cy_depth;

    // RGB camera intrinsics
    float fx_rgb, fy_rgb, cx_rgb, cy_rgb;

    // Depth-to-RGB extrinsics
    float R[9];  // row-major 3x3
    float T[3];  // translation

    // RGB-to-world extrinsics
    float R_world[9]; // row-major 3x3
    float T_world[3]; // translation

    // Filter parameters
    float max_depth_mm;
    int border_left, border_right, border_top, border_bottom;
};

// Output arrays (device pointers), pre-allocated to depth_width * depth_height
struct RgbdLoadOutputs {
    float* positions;    // 4 floats per point (x, y, z, w) in OpenGL convention
    float* normals;      // 4 floats per point (nx, ny, nz, 0)
    float* colors;       // 3 floats per point (r, g, b) in [0,1]
    float* opacities;    // 1 float per point
    float* tangents;     // 3 floats per point (tx, ty, tz) in camera space
    float* depth_cam_points; // 3 floats per point (x, y, z) in depth camera coords
    float* scale_neus;   // 1 float per point (NeuS sharpness parameter)
    int* valid_count;    // atomic counter for number of valid points (device pointer)
};

// Launch the RGBD unprojection kernel
// Returns the number of valid points (copied from device)
int launchRgbdUnprojectKernel(
    const RgbdLoadParams& params,
    RgbdLoadOutputs& outputs,
    cudaStream_t stream = 0
);

// Launch the depth covariance kernel
// Computes depth-dependent covariance for each valid point
void launchDepthCovarianceKernel(
    const float* d_depth_cam_points, // 3 floats per point
    const float* d_tangents,         // 3 floats per point
    float fx_depth, float fy_depth,
    const float* R,      // depth-to-RGB rotation (9 floats, row-major)
    const float* T_d2r,  // not used in cov, but needed for transform
    const float* R_world, // RGB-to-world rotation (9 floats, row-major)
    float* d_covX,       // 4 floats per point output
    float* d_covY,       // 4 floats per point output
    float* d_covZ,       // 4 floats per point output
    int num_points,
    cudaStream_t stream = 0
);

#endif // RGBD_LOAD_CUDA_CUH
