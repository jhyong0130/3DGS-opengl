// adam_cuda.h
#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include "adam_cuda.cuh"

#include "RenderingBase/CudaBuffer.cuh"

/// Device kernel: performs one Adam update step on each parameter element
///
/// Arguments:
///   n             - number of parameters (length of arrays)
///   params        - pointer to parameter values (device)
///   grads         - pointer to gradients (device)
///   m             - first moment buffer (device)
///   v             - second moment buffer (device)
///   beta1_pow     - beta1^t (for bias correction)
///   beta2_pow     - beta2^t
///   hparams       - hyperparameters (lr, beta1, beta2, epsilon)
__global__ void adam_update_kernel(
    int n,
    float* params,
    const float* grads,
    float* m,
    float* v,
    float beta1_pow,
    float beta2_pow,
    AdamHyperParams hparams
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load gradient
    float g = grads[idx];

    // Update biased first moment
    float m_i = m[idx];
    m_i = hparams.beta1 * m_i + (1.0f - hparams.beta1) * g;
    m[idx] = m_i;

    // Update biased second raw moment
    float v_i = v[idx];
    v_i = hparams.beta2 * v_i + (1.0f - hparams.beta2) * (g * g);
    v[idx] = v_i;

    // Compute bias-corrected moments
    float m_hat = m_i / (1.0f - beta1_pow);
    float v_hat = v_i / (1.0f - beta2_pow);

    // Parameter update
    params[idx] -= hparams.lr * m_hat / (sqrtf(v_hat) + hparams.eps);
}


// Host-side wrapper / helper
// Takes device pointers for params, grads, m, v
void adam_step_cuda(
    int n,
    GLBuffer& params,
    const float* d_grads,
    float* d_m,
    float* d_v,
    float& beta1_pow,  // this should be updated each step: beta1_pow *= beta1
    float& beta2_pow,  // similarly
    const AdamHyperParams& hparams,
    cudaStream_t stream
) {
    glFinish();

    checkCudaErrors(cudaGraphicsMapResources(1, &params.getCudaResource()));

    CudaBuffer<float> d_params = CudaBuffer<float>::fromGLBuffer(params);

    // Decide grid / block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    adam_update_kernel << < gridSize, blockSize, 0, stream >> > (
        n,
        d_params.ptr,
        d_grads,
        d_m,
        d_v,
        beta1_pow,
        beta2_pow,
        hparams
        );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();  // Check for launch error
    if (err != cudaSuccess) {
        printf("CUDA update_kernel launch error: %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &params.getCudaResource()));
}