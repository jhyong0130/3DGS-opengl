//
// Created by Briac on 27/08/2025.
//

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
#include "Backward.cuh"
#include "CVTupdate.cuh"
#include "KNN_cuda.cuh"
#include "adam_cuda.cuh"

#include <set>
#include <map>
#include <algorithm>
#include <future>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef K::Point_3 Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Vertex_iterator Vertex_iterator;
typedef Delaunay::Finite_vertices_iterator Finite_vertex_iterator;

class GaussianCloud {
public:
    bool initialized = false;
    bool shrink = false;
    bool up_sample = false;
    int nbrIter = 0;
    int nbrOptimIter = 0;
    int num_gaussians;

    float lr = 1.0f;

    // Adam hyperparameters
    AdamHyperParams hparams;

    // Variables to keep track of beta^t
    float beta1_pow;
    float beta2_pow;

    std::vector<glm::vec4> positions_cpu;
    std::vector<glm::vec4> normals_cpu;
    std::vector<glm::vec4> covX_cpu;
    std::vector<glm::vec4> covY_cpu;
    std::vector<glm::vec4> covZ_cpu;
    std::vector<glm::vec4> scales_cpu;
    std::vector<glm::vec4> rotations_cpu;
    std::vector<float> opacities_cpu;
    std::vector<float> sdf_cpu;

    // values for all the gaussians
    GLBuffer positions; // x, y, z, padding
    GLBuffer normals;
    GLBuffer covariance[3]; // 3 channel x, y, z, padding
    //GLBuffer scales;    // sx, sy, sz, padding
    //GLBuffer rotations; // rx, ry, rz, rw
    GLBuffer opacities; // alpha
    GLBuffer sdf; // alpha
    GLBuffer sh_coeffs[3]; // 3 color channels, 16 coeffs each

    // values only for the gaussians that are visible, packed tightly without gaps
    GLBuffer conic_opacity;
    GLBuffer bounding_boxes; // 2D bounding boxes of the visible gaussians
    GLBuffer eigen_vecs;
    GLBuffer predicted_colors; // view-dependent colors
    GLBuffer gaussians_indices; // indices of the visible gaussians
    GLBuffer gaussians_depths;
    GLBuffer visible_gaussians_counter; // number of visible gaussians

    GLBuffer dLoss_dpredicted_colors;
    GLBuffer dLoss_dconic_opacity;

    GLBuffer sorted_depths;
    GLBuffer sorted_gaussian_indices;

    // Triangle mesh buffers (for RGB-D depth grid triangulation)
    GLBuffer mesh_vertices;   // vec4 positions
    GLBuffer mesh_normals;    // vec4 normals
    GLBuffer mesh_colors;     // vec4 colors (RGBA)
    GLBuffer mesh_indices;    // uint indices for triangles
    int num_mesh_vertices = 0;
    int num_mesh_indices = 0;
    bool hasMesh = false;
    GLuint meshVAO = 0;

    GLuint GT_tex;
    GLuint64 GT_imageHandle = 0;


    std::vector<float*> dLoss_sh_coeffs;
    float* dLoss_SDF;

    std::vector<float*> d_m_sh_coeff; 
    std::vector<float*> d_v_sh_coeff;
    float* d_m_sdf;
    float* d_v_sdf;

    float3* pts_f3;
    uint32_t* morton_codes;
    uint32_t* sorted_indices;
    uint32_t* indices_out;
    float* distances_out;
    uint32_t* n_neighbors_out;

    float* threshold_sdf;
    unsigned char* d_flags;
    uint4* d_adjacencies;
    uint4* d_adjacencies_delaunay;
    int KVal = 32;
    int KVal_d = 0;

    int GT_cols = 0;
    int GT_rows = 0;

    lbvh_tree* knn_tree;

    Delaunay dt;
    std::vector<float4> fork_pts;
    std::future<void> delaunay_future;
    int delaunay_done = 0;

    GaussianCloud() {
        knn_tree = new lbvh_tree(32, false, false, 10.0f, true, KVal);
        hparams.lr = 1e-3f;
        hparams.beta1 = 0.9f;
        hparams.beta2 = 0.999f;
        hparams.eps = 1e-8f;

        // Variables to keep track of beta^t
        beta1_pow = hparams.beta1;
        beta2_pow = hparams.beta2;
    }

    ~GaussianCloud() {
        delete knn_tree;

        if (GT_tex) {
            glDeleteTextures(1, &GT_tex);
            GT_tex = 0;
        }
        
        if (meshVAO) {
            glDeleteVertexArrays(1, &meshVAO);
            meshVAO = 0;
        }
    }


    void initShaders();
    void GUI(Camera& camera);
    void render(Camera& camera);
    void loadGTImage(unsigned char* GT_image, int cols, int rows);
    void forward(Camera& camera, int cols, int rows);
    void backward(Camera& camera, int cols, int rows);
    void step();
    void KNN_cu();
    void updateCVT();
    void updateThresh();
    void prep_fork();
    void doDelaunay();
    void update_after_delaunay();
    void upsample(bool useCudaGLInterop);
    
    // Export triangle mesh render at a specific camera pose
    void exportCombinedMeshRenderAtPose(
        GaussianCloud& other,
        const glm::mat3& intrinsics,
        const glm::mat3& R_cam_to_world,
        const glm::vec3& T_cam_to_world,
        int width, int height,
        const std::string& outputPath);

    void exportRenderAtPose(
        const glm::mat3& intrinsics,       // Camera intrinsic matrix K (OpenCV style)
        const glm::mat3& R_cam_to_world,   // Rotation: camera to world
        const glm::vec3& T_cam_to_world,   // Translation: camera position in world
        int width, int height,
        const std::string& outputPath,
		bool useQuadRendering);             // true = quads, false = point cloud

private:
    Shader pointShader = GLShaderLoader::load("point.vs", "point.fs");
    Shader quadShader = GLShaderLoader::load("quadNeus.vs", "quadNeus.fs");
    Shader quad_interlock_Shader = GLShaderLoader::load("quad_interlock.vs", "quad_interlock.fs");
    Shader triangleMeshShader = GLShaderLoader::load("triangleMesh.vs", "triangleMesh.fs");
    Shader testVisibilityShader = GLShaderLoader::load("testVisibility.cp");
    Shader computeBoundingBoxesShader = GLShaderLoader::load("computeBoundingBoxes.cp");
    Shader predictColorsShader = GLShaderLoader::load("predict_colors.cp");
    Shader predictColorsForAllShader = GLShaderLoader::load("predict_colors_for_all.cp");

    // Backward pass
    Shader quad_interlock_bwd_Shader = GLShaderLoader::load("quad_interlock_bwd.vs", "quad_interlock_bwd.fs");

    void prepareRender(Camera& camera, bool GT = false);

    cudaGraphicsResource* GT_cudaResource;

    GLBuffer uniforms;
    FBO fbo;
    FBO emptyfbo;
    FBO fbo_gt;
    FBO emptyfbo_gt;

    FBO fbo_bwd;

    Sort sort;
    CVT cvt;
    Bwd bwd;

    int num_visible_gaussians = 0;
    bool renderAsPoints = false;
    bool renderAsQuads = false;
    bool renderAsMesh = false;
    bool renderAsTriangleMesh = false;
    float meshPointSize = 1.0f;
    float scale_modifier = 2.5f;
	float scale_neus = 1.0f;
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
        DRAW_AS_MESH,
        DRAW_AS_TRIANGLE_MESH,
        TEST_VISIBILITY,
        SORT,
        COMPUTE_BOUNDING_BOXES,
        PREDICT_COLORS_VISIBLE,
        DRAW_AS_QUADS,
        CVT_UPDATE,
        KNN,
        DELAUNAY_UPDATE,
        BLIT_FBO,
        BWD,
        NUM_OPS
    };

    QueryBuffer timers[OPERATIONS::NUM_OPS];

    GLBuffer counter;
};


#endif //HARDWARERASTERIZED3DGS_GAUSSIANCLOUD_H
