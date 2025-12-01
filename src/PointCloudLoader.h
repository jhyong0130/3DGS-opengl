//
// Created by Briac on 27/08/2025.
//

#ifndef HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H
#define HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H

#include <string>

#include "GaussianCloud.h"

class PointCloudLoader {
public:
    static void load(GaussianCloud& dst, const std::string& path, bool cudaGLInterop=true);
    static void loadRdm(GaussianCloud& dst, int nb_pts, bool cudaGLInterop = true);
    static void loadRgbd(GaussianCloud& dst, const std::string& depth_path,
        const std::string& rgb_path,
        const glm::mat3& depth_intrinsics,
        const glm::mat3& rgb_intrinsics,
        const glm::mat3& R,
        const glm::vec3& T,
        const glm::mat3& rgbToWorldR,
        const glm::vec3& rgbToWorldT,
        bool useCudaGLInterop = true);
    static void merge(GaussianCloud& dst, const GaussianCloud& a, const GaussianCloud& b, bool useCudaGLInterop);
};


#endif //HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H
