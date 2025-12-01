// adam_cuda.h
#pragma once

#include "RenderingBase/GLBuffer.h"

/// Adam hyperparameters struct
struct AdamHyperParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
};

// Host-side wrapper / helper
// Takes device pointers for params, grads, m, v
void adam_step_cuda(
    int n,
    GLBuffer&  params,
    const float* d_grads,
    float* d_m,
    float* d_v,
    float& beta1_pow,  // this should be updated each step: beta1_pow *= beta1
    float& beta2_pow,  // similarly
    const AdamHyperParams& hparams,
    cudaStream_t stream = 0
);