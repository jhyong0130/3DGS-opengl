//
// Created by Briac on 27/08/2025.
//

#ifndef HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H
#define HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H

#include <string>
#include <vector>
#include <future>
#include <memory>

#include "GaussianCloud.h"

struct RgbdFrameSequence {
    std::vector<std::string> depthPaths;
    std::vector<std::string> colorPaths;
    int currentFrame = 0;
    int totalFrames = 0;
    bool playing = false;
    float fps = 30.0f;
    double lastFrameTime = 0.0;
};

// Pre-decoded image pair (loaded on background thread, consumed on main thread).
// Uses opaque shared_ptr<void> so that cv::Mat doesn't leak into headers
// included by CUDA translation units.
struct PreloadedFrame {
    std::shared_ptr<void> depth;  // holds a cv::Mat* internally
    std::shared_ptr<void> color;  // holds a cv::Mat* internally
    bool valid = false;
};

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

    // GPU-accelerated RGBD loading (from file paths)
    static void loadRgbdGpu(GaussianCloud& dst, const std::string& depth_path,
        const std::string& rgb_path,
        const glm::mat3& depth_intrinsics,
        const glm::mat3& rgb_intrinsics,
        const glm::mat3& R,
        const glm::vec3& T,
        const glm::mat3& rgbToWorldR,
        const glm::vec3& rgbToWorldT,
        bool useCudaGLInterop = true);

    // GPU-accelerated RGBD loading (from pre-decoded PreloadedFrame — skips disk I/O)
    static void loadRgbdGpuFromMats(GaussianCloud& dst,
        const PreloadedFrame& frame,
        const glm::mat3& depth_intrinsics,
        const glm::mat3& rgb_intrinsics,
        const glm::mat3& R,
        const glm::vec3& T,
        const glm::mat3& rgbToWorldR,
        const glm::vec3& rgbToWorldT,
        bool useCudaGLInterop = true);

    static void merge(GaussianCloud& dst, const GaussianCloud& a, const GaussianCloud& b, bool useCudaGLInterop);

    // Discover all frames in a depth/color directory pair
    static RgbdFrameSequence discoverFrameSequence(
        const std::string& depthDir,
        const std::string& colorDir,
        const std::string& prefix = "frame_",
        const std::string& extension = ".png",
        int numDigits = 6);

    // Decode a depth+color image pair on a background thread
    static std::future<PreloadedFrame> preloadFrameAsync(
        const std::string& depthPath, const std::string& colorPath);

private:
    // Internal: GPU load from already-decoded images (cv::Mat passed as void*)
    static void loadRgbdGpuInternal(GaussianCloud& dst,
        const void* depth_mat, const void* rgb_mat,
        const glm::mat3& depth_intrinsics,
        const glm::mat3& rgb_intrinsics,
        const glm::mat3& R,
        const glm::vec3& T,
        const glm::mat3& rgbToWorldR,
        const glm::vec3& rgbToWorldT,
        bool useCudaGLInterop);
};


#endif //HARDWARERASTERIZED3DGS_POINTCLOUDLOADER_H
