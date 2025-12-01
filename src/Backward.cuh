//
// Created by Briac on 28/08/2025.
//

#ifndef HARDWARERASTERIZED3DGS_BWD_CUH
#define HARDWARERASTERIZED3DGS_BWD_CUH

#include "RenderingBase/GLBuffer.h"

class Bwd {
public:
    void backprop(GLBuffer& position_b, GLBuffer& sh_coeffs_red, GLBuffer& sh_coeffs_green, GLBuffer& sh_coeffs_blue, GLBuffer& covX_b, GLBuffer& covY_b, GLBuffer& covZ_b, GLBuffer& SDF_b,
                    GLBuffer& dLoss_dpredicted_colors, GLBuffer& dLoss_dconic_opacity, GLBuffer& gaussian_indices, GLBuffer& sorted_gaussian_indices,
                    float* camera_pos, float* dLoss_sh_coeffs_R, float* dLoss_sh_coeffs_G, float* dLoss_sh_coeffs_B, float* dLoss_SDF,
                    float* viewMat, float width, float height, float focal_x, float focal_y, float scale_modifier, float SDF_scale, int antialiasing, int count);
};


#endif
