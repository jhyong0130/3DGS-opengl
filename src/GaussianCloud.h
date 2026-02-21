#ifndef HARDWARERASTERIZED3DGS_GAUSSIANCLOUD_H
#define HARDWARERASTERIZED3DGS_GAUSSIANCLOUD_H

#include "RenderingBase/GLBuffer.h"
#include "RenderingBase/Shader.h"
#include "RenderingBase/GLShaderLoader.h"
#include "RenderingBase/Camera.h"
#include "RenderingBase/VAO.h"
#include "RenderingBase/GLTimer.h"
#include "RenderingBase/FBO.h"
#include "Sort.cuh"

#include <set>
#include <map>
#include <algorithm>
#include <future>

class GaussianCloud {
public:
    bool initialized = false;
    bool shrink = false;
    bool up_sample = false;
    int num_gaussians;

    std::vector<glm::vec4> positions_cpu;
    std::vector<glm::vec4> normals_cpu;
    std::vector<glm::vec4> covX_cpu;
    std::vector<glm::vec4> covY_cpu;
    std::vector<glm::vec4> covZ_cpu;
    std::vector<glm::vec4> scales_cpu;
    std::vector<glm::vec4> rotations_cpu;
    std::vector<float> opacities_cpu;
    std::vector<float> sdf_cpu;
	std::vector<float> scale_neus_cpu;

    // values for all the gaussians
    GLBuffer positions; // x, y, z, padding
    GLBuffer normals;
    GLBuffer covariance[3]; // 3 channel x, y, z, padding
    //GLBuffer scales;    // sx, sy, sz, padding
    //GLBuffer rotations; // rx, ry, rz, rw
    GLBuffer opacities; // alpha
    GLBuffer sdf; // alpha
    GLBuffer scale_neus;
    GLBuffer sh_coeffs[3]; // 3 color channels, 16 coeffs each

    // values only for the gaussians that are visible, packed tightly without gaps
    GLBuffer conic_opacity;
    GLBuffer bounding_boxes; // 2D bounding boxes of the visible gaussians
    GLBuffer eigen_vecs;
    GLBuffer predicted_colors; // view-dependent colors
    GLBuffer gaussians_indices; // indices of the visible gaussians
    GLBuffer gaussians_depths;
    GLBuffer visible_gaussians_counter; // number of visible gaussians


    GLBuffer sorted_depths;
    GLBuffer sorted_gaussian_indices;

    GaussianCloud() {}

    // Free all raw CUDA allocations (must be called before re-loading data)
    void freeRawCudaBuffers();

    // Release CPU-side vectors to save host memory after data is on GPU
    void clearCpuData();

    void initShaders();
    void GUI(Camera& camera);
    void render(Camera& camera);
    void exportRenderAtPose(
        const glm::mat3& intrinsics,
        const glm::mat3& R_cam_to_world,
        const glm::vec3& T_cam_to_world,
        int width, int height,
        const std::string& outputPath,
        bool useQuadRendering);

private:
    Shader pointShader = GLShaderLoader::load("point.vs", "point.fs");
    Shader quadShader = GLShaderLoader::load("quadNeus.vs", "quadNeus.fs");
    Shader quad_interlock_Shader = GLShaderLoader::load("quad_interlock.vs", "quad_interlock.fs");
    Shader testVisibilityShader = GLShaderLoader::load("testVisibility.cp");
    Shader computeBoundingBoxesShader = GLShaderLoader::load("computeBoundingBoxes.cp");
    Shader predictColorsShader = GLShaderLoader::load("predict_colors.cp");
    Shader predictColorsForAllShader = GLShaderLoader::load("predict_colors_for_all.cp");

    void prepareRender(Camera& camera, bool GT = false);

    GLBuffer uniforms;
    FBO fbo;
    FBO emptyfbo;

    Sort sort;

    int num_visible_gaussians = 0;
    bool renderAsPoints = false;
    bool renderAsQuads = false;
    float scale_modifier = 2.5f;
	//float scale_neus = 1.0f;
    float SDF_scale = 1.0f;
    bool antialiasing = false;
    float min_opacity = 0.02f;
    bool front_to_back = true;
    bool softwareBlending = false;
    int selected_gaussian = -1;
    int KVal_d_prev = 0;
    int fork_KVal_d = 0;
    std::vector<unsigned int> fork_ret_index;

    float thresh_vals[2]{ -0.2, 0.2 };
    int nbrIter_todo = 0;
    int nbrOptimIter_todo = 0;

    enum OPERATIONS{
        PREDICT_COLORS_ALL,
        DRAW_AS_POINTS,
        TEST_VISIBILITY,
        SORT,
        COMPUTE_BOUNDING_BOXES,
        PREDICT_COLORS_VISIBLE,
        DRAW_AS_QUADS,
        BLIT_FBO,
        NUM_OPS
    };

    QueryBuffer timers[OPERATIONS::NUM_OPS];

    GLBuffer counter;
};


#endif //HARDWARERASTERIZED3DGS_GAUSSIANCLOUD_H
