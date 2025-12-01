//
// Created by Briac on 27/08/2025.
//

#include "PointCloudLoader.h"

#include <vector>

#include "glm/vec3.hpp"
#include "glm/common.hpp"
#include "RenderingBase/VAO.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"    
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/component_wise.hpp"


#include <opencv2/opencv.hpp>
#include "miniply/miniply.h"
#include <cuda_fp16.h>

using namespace glm;

static const char *kFileTypes[] = {
        "ascii",
        "binary_little_endian",
        "binary_big_endian",
};
static const char *kPropertyTypes[] = {
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "float",
        "double",
};


static bool has_extension(const char *filename, const char *ext) {
    int j = int(strlen(ext));
    int i = int(strlen(filename)) - j;
    if (i <= 0 || filename[i - 1] != '.') {
        return false;
    }
    return strcmp(filename + i, ext) == 0;
}

// Helper function to calculate depth-dependent covariance
static void calculateDepthCovariance(const std::vector<glm::vec3>& positions,
    float fx_depth, float fy_depth,
    std::vector<glm::mat3>& covariances) {
    const float pixel_size = 3.5e-06f; // 3.5É m pixel size
    const float depth_noise_a = 0.0007f;
    const float depth_noise_b = 0.0002f;
    const float scale_factor = 1.0f;

    covariances.resize(positions.size());

    for (size_t i = 0; i < positions.size(); i++) {
        float z = fabsf(positions[i].z); // use absolute depth value

        float s_x = (pixel_size * z) / (fx_depth * pixel_size); // simplifies to z / fx_depth
        float s_y = (pixel_size * z) / (fy_depth * pixel_size); // simplifies to z / fy_depth
        float s_z = (depth_noise_a * z - depth_noise_b);
        s_z = s_z * s_z;

        float var_x = (s_x * s_x) / 4.0f;
        float var_y = (s_y * s_y) / 4.0f;
        float var_z = (s_z * s_z) / 4.0f;

        covariances[i] = glm::mat3(
            var_x * scale_factor, 0.0f, 0.0f,
            0.0f, var_y * scale_factor, 0.0f,
            0.0f, 0.0f, var_z * scale_factor
        );
    }
}

glm::vec3 computeDepthNormal(
    const cv::Mat& depth,
    int u, int v,
    float fx, float fy,
    float cx, float cy)
{
    auto getDepth = [&](int x, int y) -> float {
        if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows)
            return -1.0f;
        float d = depth.at<uint16_t>(y, x) / 1000.0f;
        return (d <= 0.0f || d > 10.0f) ? -1.0f : d;
        };

    float dz = getDepth(u, v);
    float dz_du = getDepth(u + 1, v);
    float dz_dv = getDepth(u, v + 1);

    if (dz <= 0.0f || dz_du <= 0.0f || dz_dv <= 0.0f)
        return glm::vec3(0, 0, 0);   // invalid normal

    // Convert pixels to camera coords
    glm::vec3 p((u - cx) * dz / fx, (v - cy) * dz / fy, dz);
    glm::vec3 px((u + 1 - cx) * dz_du / fx, (v - cy) * dz_du / fy, dz_du);
    glm::vec3 py((u - cx) * dz_dv / fx, (v + 1 - cy) * dz_dv / fy, dz_dv);

    glm::vec3 vx = px - p;
    glm::vec3 vy = py - p;

    glm::vec3 n = glm::normalize(glm::cross(vx, vy));

    // Ensure normal faces the camera (positive Z)
    if (n.z > 0) n = -n;

    return n;
}

glm::vec3 convertNormalToWorld(
    const glm::vec3& n_cam,
    const glm::mat3& R_depthToRgb,
    const glm::mat3& R_rgbToWorld)
{
    glm::vec3 n_rgb = R_depthToRgb * n_cam;
    glm::vec3 n_world = R_rgbToWorld * n_rgb;
    return glm::normalize(n_world);
}

bool print_ply_header(const char *filename) {
    miniply::PLYReader reader(filename);
    if (!reader.valid()) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return false;
    }

    printf("ply\n");
    printf("format %s %d.%d\n", kFileTypes[int(reader.file_type())], reader.version_major(), reader.version_minor());
    for (uint32_t i = 0, endI = reader.num_elements(); i < endI; i++) {
        const miniply::PLYElement *elem = reader.get_element(i);
        printf("element %s %u\n", elem->name.c_str(), elem->count);
        for (const miniply::PLYProperty &prop: elem->properties) {
            if (prop.countType != miniply::PLYPropertyType::None) {
                printf("property list %s %s %s\n", kPropertyTypes[uint32_t(prop.countType)],
                       kPropertyTypes[uint32_t(prop.type)], prop.name.c_str());
            } else {
                printf("property %s %s\n", kPropertyTypes[uint32_t(prop.type)], prop.name.c_str());
            }
        }
    }
    printf("end_header\n");

    while (reader.has_element()) {
        const miniply::PLYElement *elem = reader.element();
        if (elem->fixedSize || elem->count == 0) {
            reader.next_element();
            continue;
        }

        if (!reader.load_element()) {
            fprintf(stderr, "Element %s failed to load\n", elem->name.c_str());
        }
        for (const miniply::PLYProperty &prop: elem->properties) {
            if (prop.countType == miniply::PLYPropertyType::None) {
                continue;
            }
            bool mixedSize = false;
            const uint32_t firstRowCount = prop.rowCount.front();
            for (const uint32_t rowCount: prop.rowCount) {
                if (rowCount != firstRowCount) {
                    mixedSize = true;
                    break;
                }
            }
            if (mixedSize) {
                printf("Element '%s', list property '%s': not all lists have the same size\n",
                       elem->name.c_str(), prop.name.c_str());
            } else {
                printf("Element '%s', list property '%s': all lists have size %u\n",
                       elem->name.c_str(), prop.name.c_str(), firstRowCount);
            }
        }
        reader.next_element();
    }

    return true;
}


//static std::vector<float> buildPositions(aiMesh *mesh,
//                                         bool scaleToUnit) {
//    uint32_t numVertices = mesh->mNumVertices;
//
//    std::vector<float> vertices(numVertices * 3, 0.0f);
//    vec3 Vmin = vec3(+INFINITY);
//    vec3 Vmax = vec3(-INFINITY);
//
//    for (uint32_t v = 0; v < numVertices; v++) {
//        auto vertex = mesh->mVertices[v];
//        vertices[3 * v + 0] = vertex.x;
//        vertices[3 * v + 1] = vertex.y;
//        vertices[3 * v + 2] = vertex.z;
//
//        vec3 u(vertex.x, vertex.y, vertex.z);
//        Vmax = max(Vmax, u);
//        Vmin = min(Vmin, u);
//    }
//
//    if (scaleToUnit) {
//        vec3 size = Vmax - Vmin;
//        vec3 center = Vmin + size * 0.5f;
//        float half_extent = std::max(std::max(size.x, size.y), size.z) * 0.5f;
//        for (uint32_t v = 0; v < numVertices; v++) {
//            vertices[3 * v + 0] = (vertices[3 * v + 0] - center.x) / half_extent;
//            vertices[3 * v + 1] = (vertices[3 * v + 1] - center.y) / half_extent;
//            vertices[3 * v + 2] = (vertices[3 * v + 2] - center.z) / half_extent;
//        }
//    }
//
//    return vertices;
//}

float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

float h_SDF_Torus(glm::vec4 point, float R_max, float R_min) {
    float qx = sqrt(point.x * point.x + point.y * point.y) - R_max;
    float qz = point.z;
    return sqrt(qx * qx + qz * qz) - R_min;
}

float h_SDF_Sphere(glm::vec4 point, float R) {
    return sqrt(point.x * point.x + point.y * point.y + point.z * point.z) - R;
}

void PointCloudLoader::load(GaussianCloud& dst, const std::string &path, bool useCudaGLInterop) {
    dst.initialized = false;

    std::cout << "Loading point cloud: " << path << " ..." << std::endl;

    print_ply_header(path.c_str());
    std::cout << "End of header."<< std::endl;

    miniply::PLYReader reader(path.c_str());
    if (!reader.valid()) {
        std::cout << "Couldn't read " <<path << std::endl;
        return;
    }

    assert(reader.has_element());

    const miniply::PLYElement *elem = reader.element();

    if (!reader.load_element()) {
        std::cout <<"Element" <<elem->name <<" failed to load." <<std::endl;
        return;
    }

    assert(elem->name == "vertex");

    dst.num_gaussians = (int)elem->count;

    const uint pos_idx[3] = {
            elem->find_property("x"),
            elem->find_property("y"),
            elem->find_property("z")
    };
    const uint rot_idx[4] = {
            elem->find_property("rot_0"),
            elem->find_property("rot_1"),
            elem->find_property("rot_2"),
            elem->find_property("rot_3")
    };
    const uint scale_idx[3] = {
            elem->find_property("scale_0"),
            elem->find_property("scale_1"),
            elem->find_property("scale_2")
    };
    const uint opacity_idx[1] = {
            elem->find_property("opacity")
    };

    dst.positions_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    for(int i=0; i<dst.num_gaussians; i++){
        dst.positions_cpu[i].w = 1.0f;
    }
    reader.extract_properties_with_stride(pos_idx, 3, miniply::PLYPropertyType::Float, dst.positions_cpu.data(), 4*sizeof(float));
    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, false, true);

    //dst.scales_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    //reader.extract_properties_with_stride(scale_idx, 3, miniply::PLYPropertyType::Float, dst.scales_cpu.data(), 4*sizeof(float));
    //for(int i=0; i<dst.num_gaussians; i++){
    //    dst.scales_cpu[i] = exp(dst.scales_cpu[i]); // apply exponential activation
    //    dst.scales_cpu[i].w = 0.0f;
    //}
    //dst.scales.storeData(dst.scales_cpu.data(), dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, false, true);

    //dst.rotations_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    //reader.extract_properties(rot_idx, 4, miniply::PLYPropertyType::Float, dst.rotations_cpu.data());
    //dst.rotations.storeData(dst.rotations_cpu.data(), dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, false, true);

    dst.opacities_cpu = std::vector<float>(dst.num_gaussians);
    reader.extract_properties(opacity_idx, 1, miniply::PLYPropertyType::Float, dst.opacities_cpu.data());
    for(int i=0; i<dst.num_gaussians; i++){
        dst.opacities_cpu[i] = sigmoid(dst.opacities_cpu[i]); // apply sigmoid activation
    }
    dst.opacities.storeData(dst.opacities_cpu.data(), dst.num_gaussians, 1*sizeof(float), 0, useCudaGLInterop, false, true);

    uint sh_idx[48];
    for(int i=0; i<48; i++){
        const std::string prop_name = i < 3 ? "f_dc_" + std::to_string(i) : "f_rest_" + std::to_string(i-3);
        sh_idx[i] = elem->find_property(prop_name.c_str());
    }

    for(int i=0; i<3; i++) {
        float* sh_coeffs = new float[dst.num_gaussians * 16];
        uint channel_idx[16];
        for(int j=0; j<16; j++){
//            channel_idx[j] = sh_idx[j*3+i];
            if(j == 0){
                channel_idx[0] = sh_idx[i];
            }else{
                channel_idx[j] = sh_idx[3+i*15+j-1];
            }
        }
        reader.extract_properties(channel_idx, 16, miniply::PLYPropertyType::Float, sh_coeffs);
        dst.sh_coeffs[i].storeData(sh_coeffs, dst.num_gaussians, 16*sizeof(float), 0, useCudaGLInterop, false, true);
        delete[] sh_coeffs;
    }

    dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
    dst.gaussians_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.gaussians_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);
    dst.sorted_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.sorted_gaussian_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);

    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);

    dst.initialized = true;

    std::cout << "Finished loading point cloud." << std::endl;
}


void PointCloudLoader::loadRdm(GaussianCloud& dst, int nb_pts, bool useCudaGLInterop) {
    dst.initialized = false;

    std::cout << "Loading random point cloud with " << nb_pts << " points ..." << std::endl;

    assert(nb_pts > 0);

    dst.num_gaussians = (int)nb_pts;

    dst.positions_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    for (int i = 0; i < dst.num_gaussians; i++) {
        dst.positions_cpu[i].x = 1.0f - 2.0f * static_cast<float>(rand()) / RAND_MAX;
        dst.positions_cpu[i].y = 1.0f - 2.0f * static_cast<float>(rand()) / RAND_MAX;
        dst.positions_cpu[i].z = 1.0f - 2.0f * static_cast<float>(rand()) / RAND_MAX; // 1.0f; //
        dst.positions_cpu[i].w = 1.0f;
    }
    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    //dst.scales_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    //for (int i = 0; i < dst.num_gaussians; i++) {
    //   dst.scales_cpu[i] = glm::vec4(0.05f);// exp(glm::vec4(1.0f)); // apply exponential activation
    //    dst.scales_cpu[i].w = 0.0f;
    //}
    //dst.scales.storeData(dst.scales_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    //dst.rotations_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    //for (int i = 0; i < dst.num_gaussians; i++) {
    //    dst.rotations_cpu[i] = glm::vec4(0.0f); // apply exponential activation
    //    dst.rotations_cpu[i].w = 1.0f;
    //}
    //dst.rotations.storeData(dst.rotations_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    dst.sdf_cpu = std::vector<float>(dst.num_gaussians);
    dst.opacities_cpu = std::vector<float>(dst.num_gaussians);
    for (int i = 0; i < dst.num_gaussians; i++) {
        float sdf_val = h_SDF_Sphere(dst.positions_cpu[i], 0.5f); // h_SDF_Torus(dst.positions_cpu[i], 0.6f, 0.4f);
        dst.sdf_cpu[i] = sdf_val;
        dst.opacities_cpu[i] = sigmoid(exp(-sdf_val * sdf_val/0.01f)); // apply sigmoid activation sdf[i] = h_SDF_Torus(pts[i], 0.6f, 0.4f);
    }
    dst.sdf.storeData(dst.sdf_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);
    dst.opacities.storeData(dst.opacities_cpu.data(), dst.num_gaussians, 1 * sizeof(float), 0, useCudaGLInterop, false, true);

    for (int i = 0; i < 3; i++) {
        float* sh_coeffs = new float[dst.num_gaussians * 16];
        for (int k = 0; k < dst.num_gaussians; k++) {
            for (int j = 0; j < 16; j++) {
                //            channel_idx[j] = sh_idx[j*3+i];
                if (j == 0) {
                    if (i == 0)
                        sh_coeffs[k * 16 + 0] = 0.5f;//fabs(dst.positions_cpu[k].x);// 1.0f;// sh_idx[i];
                    if (i == 1)
                        sh_coeffs[k * 16 + 0] = 0.5f;//fabs(dst.positions_cpu[k].y);// 1.0f;// sh_idx[i];
                    if (i == 2)
                        sh_coeffs[k * 16 + 0] = 0.5f;//fabs(dst.positions_cpu[k].z);// 1.0f;// sh_idx[i];
                }
                else {
                    sh_coeffs[k * 16 + j] = 0.0f;// sh_idx[3 + i * 15 + j - 1];
                }
            }
        }
        dst.sh_coeffs[i].storeData(sh_coeffs, dst.num_gaussians, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
        delete[] sh_coeffs;
    }

    for (int i = 0; i < 3; i++) {
        float* cov = new float[dst.num_gaussians * 4];
        for (int k = 0; k < dst.num_gaussians; k++) {
            for (int j = 0; j < 4; j++) {
                cov[k * 4 + j] = 0.0;
            }
            cov[k * 4 + i] = 0.01f;
        }
        dst.covariance[i].storeData(cov, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    }
    /*for (int i = 0; i < 3; i++) {
        dst.covariance[i].storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    }*/
    //dst.sdf.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);

    dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
    dst.gaussians_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.gaussians_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);
    dst.sorted_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.sorted_gaussian_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);

    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);

    dst.dLoss_dpredicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(__half), 0, useCudaGLInterop, true, true);
    dst.dLoss_dconic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(__half), 0, useCudaGLInterop, true, true);

    cudaMalloc((void**)&dst.threshold_sdf, 2 * sizeof(float));
    cudaMalloc((void**)&dst.d_flags, dst.num_gaussians * sizeof(unsigned char));
    cudaMalloc((void**)&dst.d_adjacencies, (dst.KVal + dst.KVal_d) * dst.num_gaussians * sizeof(uint));
    cudaMalloc((void**)&dst.d_adjacencies_delaunay, dst.KVal_d * dst.num_gaussians * sizeof(uint));
    
    cudaMalloc((void**)&dst.pts_f3, 3 * dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.morton_codes, dst.num_gaussians * sizeof(uint64_t));
    cudaMalloc((void**)&dst.sorted_indices, dst.num_gaussians * sizeof(uint32_t));
    cudaMalloc((void**)&dst.indices_out, dst.KVal * dst.num_gaussians * sizeof(uint32_t));
    cudaMalloc((void**)&dst.distances_out, dst.KVal * dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.n_neighbors_out, dst.num_gaussians * sizeof(uint32_t));

    for (int k = 0; k < 3; k++) {
        float* tmp;
        float* d_m_tmp;
        float* d_v_tmp;
        cudaMalloc((void**)&tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMalloc((void**)&d_m_tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMalloc((void**)&d_v_tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMemset(d_m_tmp, 0, 16 * dst.num_gaussians * sizeof(float));
        cudaMemset(d_v_tmp, 0, 16 * dst.num_gaussians * sizeof(float));
        dst.dLoss_sh_coeffs.push_back(tmp);
        dst.d_m_sh_coeff.push_back(d_m_tmp);
        dst.d_v_sh_coeff.push_back(d_v_tmp);
    }

    cudaMalloc((void**)&dst.dLoss_SDF, dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.d_m_sdf, dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.d_v_sdf, dst.num_gaussians * sizeof(float));
    cudaMemset(dst.d_m_sdf, 0, dst.num_gaussians * sizeof(float));
    cudaMemset(dst.d_v_sdf, 0, dst.num_gaussians * sizeof(float));

    dst.fork_pts.resize(dst.num_gaussians);

    dst.initialized = true;

    std::cout << "Finished loading random point cloud." << std::endl;
}

void PointCloudLoader::loadRgbd(GaussianCloud& dst, const std::string& depth_path,
    const std::string& rgb_path,
    const glm::mat3& depth_intrinsics,
    const glm::mat3& rgb_intrinsics,
    const glm::mat3& R,
    const glm::vec3& T,
    const glm::mat3& rgbToWorldR,
	const glm::vec3& rgbToWorldT,
    bool useCudaGLInterop
) {
    dst.initialized = false;

    std::cout << "Loading RGBD point cloud from depth: " << depth_path
        << " and RGB: " << rgb_path << " ..." << std::endl;

    // Load depth and RGB images using OpenCV
    cv::Mat depth_image = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
    cv::Mat rgb_image = cv::imread(rgb_path, cv::IMREAD_COLOR);

    if (depth_image.empty() || rgb_image.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return;
    }

    // Extract camera intrinsics
    float FX_DEPTH = depth_intrinsics[0][0];
    float FY_DEPTH = depth_intrinsics[1][1];
    float CX_DEPTH = depth_intrinsics[2][0];
    float CY_DEPTH = depth_intrinsics[2][1];

    float FX_RGB = rgb_intrinsics[0][0];
    float FY_RGB = rgb_intrinsics[1][1];
    float CX_RGB = rgb_intrinsics[2][0];
    float CY_RGB = rgb_intrinsics[2][1];

    int h_d = depth_image.rows;
    int w_d = depth_image.cols;
    int h_rgb = rgb_image.rows;
    int w_rgb = rgb_image.cols;

    // First pass: count valid points
    std::vector<glm::vec4> temp_positions;
    std::vector<glm::vec3> temp_colors;
    std::vector<glm::vec3> depth_cam_points;
    std::vector<glm::vec4> temp_normals;

    for (int i = 0; i < h_d; i++) {
        for (int j = 0; j < w_d; j++) {
            // Get depth value (assuming depth is in mm, convert to meters)
            float depth = depth_image.at<uint16_t>(i, j) / 1000.0f;

            // Skip invalid depth values
            if (depth <= 0.0f || depth > 10.0f) continue;

            // Convert pixel coordinates to 3D point in depth camera coordinates
            float x_d = (j - CX_DEPTH) * depth / FX_DEPTH;
            float y_d = (i - CY_DEPTH) * depth / FY_DEPTH;
            float z_d = depth;

            // Transform to RGB camera coordinates using R and T
            glm::vec3 point_depth = glm::vec3(x_d, y_d, z_d);
            glm::vec3 point_rgb = R * point_depth + T;

            float x_rgb = point_rgb.x;
            float y_rgb = point_rgb.y;
            float z_rgb = point_rgb.z;

            // Skip points behind the RGB camera
            if (z_rgb <= 0.0f) continue;

            // Project to RGB image coordinates
            float u = (x_rgb * FX_RGB) / z_rgb + CX_RGB;
            float v = (y_rgb * FY_RGB) / z_rgb + CY_RGB;

            // Check if projection is within RGB image bounds
            if (u < 0 || u >= w_rgb || v < 0 || v >= h_rgb) continue;

            // Clip to RGB image bounds
            int u_int = std::max(0, std::min(w_rgb - 1, (int)round(u)));
            int v_int = std::max(0, std::min(h_rgb - 1, (int)round(v)));

            // Get RGB color values (BGR format, normalize to [0,1])
            cv::Vec3b color = rgb_image.at<cv::Vec3b>(v_int, u_int);

            // Transform to world coordinates using rotation and translation
            //glm::vec3 pos_world3 = glm::transpose(rgbToWorldR) * (point_rgb - rgbToWorldT) ;
            glm::vec3 pos_world3 = glm::transpose(rgbToWorldR) * point_rgb + rgbToWorldT;
            glm::vec4 pos_world_h = glm::vec4(pos_world3, 1.0f);

			// Compute normals
            glm::vec3 normal_cam = computeDepthNormal(
                depth_image, j, i,
                FX_DEPTH, FY_DEPTH,
                CX_DEPTH, CY_DEPTH
            );

            if (normal_cam == glm::vec3(0)) continue; // skip invalid normals

            glm::vec3 normal_world = convertNormalToWorld(
                normal_cam,
                R,                // depth Å® RGB rotation
                rgbToWorldR       // RGB Å® world rotation
            );
			glm::vec4 normal_world_h = glm::vec4(normal_world, 0.0f);

            temp_positions.push_back(pos_world_h);
            temp_normals.push_back(normal_world_h);
            temp_colors.push_back(glm::vec3(color[2] / 255.0f, color[1] / 255.0f, color[0] / 255.0f));
			depth_cam_points.push_back(point_depth); // Store depth camera points for covariance calculation
        }
    }

    dst.num_gaussians = (int)temp_positions.size();

    if (dst.num_gaussians == 0) {
        std::cerr << "Error: No valid points generated from RGBD images!" << std::endl;
        return;
    }

    std::cout << "Generated " << dst.num_gaussians << " points from RGBD images" << std::endl;

    dst.scales_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    for (int i = 0; i < dst.num_gaussians; i++) {
        dst.scales_cpu[i] = glm::vec4(0.05f);// exp(glm::vec4(1.0f)); // apply exponential activation
        dst.scales_cpu[i].w = 0.0f;
    }
    dst.scales.storeData(dst.scales_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    dst.rotations_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    for (int i = 0; i < dst.num_gaussians; i++) {
        dst.rotations_cpu[i] = glm::vec4(0.0f); // apply exponential activation
        dst.rotations_cpu[i].w = 1.0f;
    }
    dst.rotations.storeData(dst.rotations_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Store positions
    dst.positions_cpu = temp_positions;
    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

	// Store normals
	dst.normals_cpu = temp_normals;
	dst.normals.storeData(dst.normals_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Initialize opacities to 1.0 for valid points
    dst.opacities_cpu = std::vector<float>(dst.num_gaussians, 1.0f);
    dst.opacities.storeData(dst.opacities_cpu.data(), dst.num_gaussians, 1 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Store RGB colors as spherical harmonics coefficients (DC component only)
    const float SH_C0 = 0.28209479177387814f;
    for (int i = 0; i < 3; i++) {
        float* sh_coeffs = new float[dst.num_gaussians * 16];
        for (int k = 0; k < dst.num_gaussians; k++) {
            // Convert RGB [0,1] to SH coefficient
            // Formula: SH_DC = (RGB - 0.5) / C0
            sh_coeffs[k * 16 + 0] = (temp_colors[k][i] - 0.5f) / SH_C0;

            // Set remaining coefficients to 0
            for (int j = 1; j < 16; j++) {
                sh_coeffs[k * 16 + j] = 0.0f;
            }
        }
        dst.sh_coeffs[i].storeData(sh_coeffs, dst.num_gaussians, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
        delete[] sh_coeffs;
    }

    // Calculate depth-dependent covariance matrices
    std::vector<glm::mat3> covariance_matrices;
    calculateDepthCovariance(depth_cam_points, FX_DEPTH, FY_DEPTH, covariance_matrices);

    std::cout << "Calculated depth-dependent covariances for " << covariance_matrices.size() << " points" << std::endl;

   // Store covariance matrices
    dst.covX_cpu.resize(dst.num_gaussians);
    dst.covY_cpu.resize(dst.num_gaussians);
    dst.covZ_cpu.resize(dst.num_gaussians);
    for (int k = 0; k < dst.num_gaussians; k++) {
        const glm::mat3& C = covariance_matrices[k];
        dst.covX_cpu[k] = glm::vec4(C[0][0], C[0][1], C[0][2], 0.0f);
        dst.covY_cpu[k] = glm::vec4(C[1][0], C[1][1], C[1][2], 0.0f);
        dst.covZ_cpu[k] = glm::vec4(C[2][0], C[2][1], C[2][2], 0.0f);
    }
    // Upload to GPU rows
    for (int row = 0; row < 3; row++) {
        float* cov = new float[dst.num_gaussians * 4];
        for (int k = 0; k < dst.num_gaussians; k++) {
            const glm::vec4& src = (row == 0 ? dst.covX_cpu[k] : (row == 1 ? dst.covY_cpu[k] : dst.covZ_cpu[k]));
            cov[k * 4 + 0] = src.x;
            cov[k * 4 + 1] = src.y;
            cov[k * 4 + 2] = src.z;
            cov[k * 4 + 3] = 0.0f;
        }
        dst.covariance[row].storeData(cov, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
        delete[] cov;
    }

    // Initialize remaining buffers
    dst.sdf_cpu = std::vector<float>(dst.num_gaussians, 0.0f);
    dst.sdf.storeData(dst.sdf_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

    dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
    dst.gaussians_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.gaussians_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);
    dst.sorted_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.sorted_gaussian_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);

    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);

    dst.dLoss_dpredicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(__half), 0, useCudaGLInterop, true, true);
    dst.dLoss_dconic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(__half), 0, useCudaGLInterop, true, true);

    cudaMalloc((void**)&dst.threshold_sdf, 2 * sizeof(float));
    cudaMalloc((void**)&dst.d_flags, dst.num_gaussians * sizeof(unsigned char));
    cudaMalloc((void**)&dst.d_adjacencies, (dst.KVal + dst.KVal_d) * dst.num_gaussians * sizeof(uint));
    cudaMalloc((void**)&dst.d_adjacencies_delaunay, dst.KVal_d * dst.num_gaussians * sizeof(uint));

    cudaMalloc((void**)&dst.pts_f3, 3 * dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.morton_codes, dst.num_gaussians * sizeof(uint64_t));
    cudaMalloc((void**)&dst.sorted_indices, dst.num_gaussians * sizeof(uint32_t));
    cudaMalloc((void**)&dst.indices_out, dst.KVal * dst.num_gaussians * sizeof(uint32_t));
    cudaMalloc((void**)&dst.distances_out, dst.KVal * dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.n_neighbors_out, dst.num_gaussians * sizeof(uint32_t));

    for (int k = 0; k < 3; k++) {
        float* tmp;
        float* d_m_tmp;
        float* d_v_tmp;
        cudaMalloc((void**)&tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMalloc((void**)&d_m_tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMalloc((void**)&d_v_tmp, 16 * dst.num_gaussians * sizeof(float));
        cudaMemset(d_m_tmp, 0, 16 * dst.num_gaussians * sizeof(float));
        cudaMemset(d_v_tmp, 0, 16 * dst.num_gaussians * sizeof(float));
        dst.dLoss_sh_coeffs.push_back(tmp);
        dst.d_m_sh_coeff.push_back(d_m_tmp);
        dst.d_v_sh_coeff.push_back(d_v_tmp);
    }

    cudaMalloc((void**)&dst.dLoss_SDF, dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.d_m_sdf, dst.num_gaussians * sizeof(float));
    cudaMalloc((void**)&dst.d_v_sdf, dst.num_gaussians * sizeof(float));
    cudaMemset(dst.d_m_sdf, 0, dst.num_gaussians * sizeof(float));
    cudaMemset(dst.d_v_sdf, 0, dst.num_gaussians * sizeof(float));

    dst.fork_pts.resize(dst.num_gaussians);

    dst.initialized = true;

    std::cout << "Finished loading RGBD point cloud." << std::endl;
}

void PointCloudLoader::merge(GaussianCloud& dst,
    const GaussianCloud& a,
    const GaussianCloud& b,
    bool useCudaGLInterop)
{
    // Caller should have called dst.initShaders() once before first merge.
    const bool hasA = a.initialized && a.num_gaussians > 0;
    const bool hasB = b.initialized && b.num_gaussians > 0;

    if (!hasA && !hasB) {
        dst.initialized = false;
        dst.num_gaussians = 0;
        std::cout << "Merge: no sources initialized.\n";
        return;
    }

    const int nA = hasA ? a.num_gaussians : 0;
    const int nB = hasB ? b.num_gaussians : 0;
    dst.num_gaussians = nA + nB;

    // Positions
    dst.positions_cpu.clear();
    dst.positions_cpu.reserve(dst.num_gaussians);
    if (hasA) dst.positions_cpu.insert(dst.positions_cpu.end(), a.positions_cpu.begin(), a.positions_cpu.end());
    if (hasB) dst.positions_cpu.insert(dst.positions_cpu.end(), b.positions_cpu.begin(), b.positions_cpu.end());
    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Opacities
    dst.opacities_cpu.clear();
    dst.opacities_cpu.reserve(dst.num_gaussians);
    if (hasA) dst.opacities_cpu.insert(dst.opacities_cpu.end(), a.opacities_cpu.begin(), a.opacities_cpu.end());
    if (hasB) dst.opacities_cpu.insert(dst.opacities_cpu.end(), b.opacities_cpu.begin(), b.opacities_cpu.end());
    if ((int)dst.opacities_cpu.size() != dst.num_gaussians) {
        // Fill missing with 1.0f if some source had no CPU copy
        const int missing = dst.num_gaussians - (int)dst.opacities_cpu.size();
        dst.opacities_cpu.insert(dst.opacities_cpu.end(), missing, 1.0f);
    }
    dst.opacities.storeData(dst.opacities_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

    // SDF (if missing, zeros)
    dst.sdf_cpu.clear();
    if (hasA) {
        if (!a.sdf_cpu.empty()) dst.sdf_cpu.insert(dst.sdf_cpu.end(), a.sdf_cpu.begin(), a.sdf_cpu.end());
        else dst.sdf_cpu.insert(dst.sdf_cpu.end(), nA, 0.0f);
    }
    if (hasB) {
        if (!b.sdf_cpu.empty()) dst.sdf_cpu.insert(dst.sdf_cpu.end(), b.sdf_cpu.begin(), b.sdf_cpu.end());
        else dst.sdf_cpu.insert(dst.sdf_cpu.end(), nB, 0.0f);
    }
    dst.sdf.storeData(dst.sdf_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

    // SH coeffs: 16 floats per channel
    auto fetchSH = [](const GaussianCloud& g, int ch)->std::vector<float> {
        if (!g.initialized || g.num_gaussians == 0) return {};
        return const_cast<GLBuffer&>(g.sh_coeffs[ch]).getAsFloats(16 * g.num_gaussians);
        };
    for (int ch = 0; ch < 3; ++ch) {
        std::vector<float> merged(16 * dst.num_gaussians);
        size_t offset = 0;
        if (hasA) {
            auto sh = fetchSH(a, ch);
            if (!sh.empty()) {
                std::memcpy(merged.data() + offset, sh.data(), sh.size() * sizeof(float));
            }
            else {
                // fallback: DC=0.5, others 0
                for (int i = 0; i < nA; i++) { merged[offset + i * 16 + 0] = 0.5f; for (int l = 1; l < 16; l++) merged[offset + i * 16 + l] = 0.0f; }
            }
            offset += 16 * nA;
        }
        if (hasB) {
            auto sh = fetchSH(b, ch);
            if (!sh.empty()) {
                std::memcpy(merged.data() + offset, sh.data(), sh.size() * sizeof(float));
            }
            else {
                for (int i = 0; i < nB; i++) { merged[offset + i * 16 + 0] = 0.5f; for (int l = 1; l < 16; l++) merged[offset + i * 16 + l] = 0.0f; }
            }
            offset += 16 * nB;
        }
        dst.sh_coeffs[ch].storeData(merged.data(), dst.num_gaussians, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
    }

    // Merge normals, rotations and scales robustly:
    auto readVec4FromGLBuffer = [](const GLBuffer& buf, int count)->std::vector<glm::vec4> {
        if (count == 0) return {};
        auto floats = const_cast<GLBuffer&>(buf).getAsFloats(count * 4);
        if (floats.empty()) return {};
        std::vector<glm::vec4> out(count);
        for (int i = 0; i < count; ++i) {
            out[i] = glm::vec4(floats[4 * i + 0], floats[4 * i + 1], floats[4 * i + 2], floats[4 * i + 3]);
        }
        return out;
        };

    // Normals
    dst.normals_cpu.clear();
    dst.normals_cpu.reserve(dst.num_gaussians);
    if (hasA && !a.normals_cpu.empty()) {
        dst.normals_cpu.insert(dst.normals_cpu.end(), a.normals_cpu.begin(), a.normals_cpu.end());
    }
    else if (hasA && a.initialized) {
        auto nA_cpu = readVec4FromGLBuffer(a.normals, nA);
        if (!nA_cpu.empty()) dst.normals_cpu.insert(dst.normals_cpu.end(), nA_cpu.begin(), nA_cpu.end());
    }
    if (hasB && !b.normals_cpu.empty()) {
        dst.normals_cpu.insert(dst.normals_cpu.end(), b.normals_cpu.begin(), b.normals_cpu.end());
    }
    else if (hasB && b.initialized) {
        auto nB_cpu = readVec4FromGLBuffer(b.normals, nB);
        if (!nB_cpu.empty()) dst.normals_cpu.insert(dst.normals_cpu.end(), nB_cpu.begin(), nB_cpu.end());
    }
    // Fill missing normals with (0,0,1,0)
    if ((int)dst.normals_cpu.size() != dst.num_gaussians) {
        int missing = dst.num_gaussians - (int)dst.normals_cpu.size();
        for (int i = 0; i < missing; ++i) dst.normals_cpu.push_back(glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    }
    dst.normals.storeData(dst.normals_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Rotations (quaternions)
    dst.rotations_cpu.clear();
    dst.rotations_cpu.reserve(dst.num_gaussians);
    if (hasA && !a.rotations_cpu.empty()) {
        dst.rotations_cpu.insert(dst.rotations_cpu.end(), a.rotations_cpu.begin(), a.rotations_cpu.end());
    }
    else if (hasA && a.initialized) {
        auto rA_cpu = readVec4FromGLBuffer(a.rotations, nA);
        if (!rA_cpu.empty()) dst.rotations_cpu.insert(dst.rotations_cpu.end(), rA_cpu.begin(), rA_cpu.end());
    }
    if (hasB && !b.rotations_cpu.empty()) {
        dst.rotations_cpu.insert(dst.rotations_cpu.end(), b.rotations_cpu.begin(), b.rotations_cpu.end());
    }
    else if (hasB && b.initialized) {
        auto rB_cpu = readVec4FromGLBuffer(b.rotations, nB);
        if (!rB_cpu.empty()) dst.rotations_cpu.insert(dst.rotations_cpu.end(), rB_cpu.begin(), rB_cpu.end());
    }
    if ((int)dst.rotations_cpu.size() != dst.num_gaussians) {
        // default to identity quaternion (0,0,0,1)
        int missing = dst.num_gaussians - (int)dst.rotations_cpu.size();
        for (int i = 0; i < missing; ++i) dst.rotations_cpu.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    dst.rotations.storeData(dst.rotations_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Scales
    dst.scales_cpu.clear();
    dst.scales_cpu.reserve(dst.num_gaussians);
    if (hasA && !a.scales_cpu.empty()) {
        dst.scales_cpu.insert(dst.scales_cpu.end(), a.scales_cpu.begin(), a.scales_cpu.end());
    }
    else if (hasA && a.initialized) {
        auto sA_cpu = readVec4FromGLBuffer(a.scales, nA);
        if (!sA_cpu.empty()) dst.scales_cpu.insert(dst.scales_cpu.end(), sA_cpu.begin(), sA_cpu.end());
    }
    if (hasB && !b.scales_cpu.empty()) {
        dst.scales_cpu.insert(dst.scales_cpu.end(), b.scales_cpu.begin(), b.scales_cpu.end());
    }
    else if (hasB && b.initialized) {
        auto sB_cpu = readVec4FromGLBuffer(b.scales, nB);
        if (!sB_cpu.empty()) dst.scales_cpu.insert(dst.scales_cpu.end(), sB_cpu.begin(), sB_cpu.end());
    }
    if ((int)dst.scales_cpu.size() != dst.num_gaussians) {
        int missing = dst.num_gaussians - (int)dst.scales_cpu.size();
        for (int i = 0; i < missing; ++i) dst.scales_cpu.push_back(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
    }
    dst.scales.storeData(dst.scales_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    if (hasA || hasB) {
        dst.covX_cpu.clear(); dst.covY_cpu.clear(); dst.covZ_cpu.clear();
        dst.covX_cpu.reserve(dst.num_gaussians);
        dst.covY_cpu.reserve(dst.num_gaussians);
        dst.covZ_cpu.reserve(dst.num_gaussians);

        if (hasA) {
            if (!a.covX_cpu.empty()) dst.covX_cpu.insert(dst.covX_cpu.end(), a.covX_cpu.begin(), a.covX_cpu.end());
            if (!a.covY_cpu.empty()) dst.covY_cpu.insert(dst.covY_cpu.end(), a.covY_cpu.begin(), a.covY_cpu.end());
            if (!a.covZ_cpu.empty()) dst.covZ_cpu.insert(dst.covZ_cpu.end(), a.covZ_cpu.begin(), a.covZ_cpu.end());
        }
        if (hasB) {
            if (!b.covX_cpu.empty()) dst.covX_cpu.insert(dst.covX_cpu.end(), b.covX_cpu.begin(), b.covX_cpu.end());
            if (!b.covY_cpu.empty()) dst.covY_cpu.insert(dst.covY_cpu.end(), b.covY_cpu.begin(), b.covY_cpu.end());
            if (!b.covZ_cpu.empty()) dst.covZ_cpu.insert(dst.covZ_cpu.end(), b.covZ_cpu.begin(), b.covZ_cpu.end());
        }

        // Correct fallback per row
        auto ensureCovSize = [&](std::vector<glm::vec4>& v, int row) {
            if ((int)v.size() != dst.num_gaussians) {
                int missing = dst.num_gaussians - (int)v.size();
                for (int i = 0; i < missing; i++) {
                    if (row == 0) v.push_back(glm::vec4(0.01f, 0.0f, 0.0f, 0.0f));
                    else if (row == 1) v.push_back(glm::vec4(0.0f, 0.01f, 0.0f, 0.0f));
                    else v.push_back(glm::vec4(0.0f, 0.0f, 0.01f, 0.0f));
                }
            }
            };
        ensureCovSize(dst.covX_cpu, 0);
        ensureCovSize(dst.covY_cpu, 1);
        ensureCovSize(dst.covZ_cpu, 2);

        for (int row = 0; row < 3; row++) {
            float* cov = new float[dst.num_gaussians * 4];
            for (int k = 0; k < dst.num_gaussians; k++) {
                const glm::vec4& src = (row == 0 ? dst.covX_cpu[k] : (row == 1 ? dst.covY_cpu[k] : dst.covZ_cpu[k]));
                cov[k * 4 + 0] = src.x;
                cov[k * 4 + 1] = src.y;
                cov[k * 4 + 2] = src.z;
                cov[k * 4 + 3] = 0.0f;
            }
            dst.covariance[row].storeData(cov, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
            delete[] cov;
        }
    }

    // Derived buffers for rendering pipeline
    dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
    dst.gaussians_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.gaussians_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);
    dst.sorted_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.sorted_gaussian_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);

    dst.scales_cpu.assign(dst.num_gaussians, glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
    dst.scales.storeData(dst.scales_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    dst.rotations_cpu.assign(dst.num_gaussians, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    dst.rotations.storeData(dst.rotations_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);

    dst.initialized = true;

    std::cout << "Merged clouds: " << nA << " + " << nB
        << " = " << dst.num_gaussians << " gaussians.\n";
}