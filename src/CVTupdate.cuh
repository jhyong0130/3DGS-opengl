#ifndef CVTSPLAT_CVTUPDATE_CUH
#define CVTSPLAT_CVTUPDATE_CUH

#include "RenderingBase/GLBuffer.h"

class CVT {
public:
    void update(GLBuffer& sdf, GLBuffer& vertices, uint4* adjacents, GLBuffer& covMatrixX, GLBuffer& covMatrixY, GLBuffer& covMatrixZ, unsigned char* flag, int K_val, float* thresh_vals, int count);

    void cpy_pts(GLBuffer& positions, float3* pts_f3, int count);

    void min_lvls(GLBuffer& sdf, float* threshold_sdf, float* min_lvl, int count);

    void map_to_cpu(GLBuffer& vertices, float4* data, int count);

    void map_to_CUDA(GLBuffer& sdf, GLBuffer& vertices, GLBuffer& covMatrixX, GLBuffer& covMatrixY, GLBuffer& covMatrixZ,
        GLBuffer& sh_coeffsR, GLBuffer& sh_coeffsG, GLBuffer& sh_coeffsB,
        float* sdf_in, float4* vertices_in,
        float4* covMatrixX_in, float4* covMatrixY_in, float4* covMatrixZ_in,
        float* sh_coeffsR_in, float* sh_coeffsG_in, float* sh_coeffsB_in, int num_gaussians);
};

#endif 