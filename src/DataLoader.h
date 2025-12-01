//
// Created by Briac on 27/08/2025.
//

#ifndef HARDWARERASTERIZED3DGS_DATALOADER_H
#define HARDWARERASTERIZED3DGS_DATALOADER_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "GaussianCloud.h"
#include "RenderingBase/Camera.h"

class DataLoader {
public:
    std::vector<unsigned char*> images_gpu;
    std::vector<glm::mat4> poses;
    Camera camera;

    int cols = 0;
    int rows = 0;

    void load(const std::string& folder);
};


#endif //HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H
