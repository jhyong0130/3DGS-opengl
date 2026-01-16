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

// Helper function to calculate depth-dependent covariance (all units in mm)
static void calculateDepthCovariance(const std::vector<glm::vec3>& positions,
    float fx_depth, float fy_depth,
    std::vector<glm::mat3>& covariances) {
    // Noise model parameters adjusted for mm units
    // depth_noise_a and depth_noise_b are relative to depth in mm
    const float depth_noise_a = 0.0007f;   // dimensionless coefficient
    const float depth_noise_b = 0.2f;      // in mm (was 0.0002m = 0.2mm)
    const float scale_factor = 1.0f;

    covariances.resize(positions.size());

    for (size_t i = 0; i < positions.size(); i++) {
        // Use absolute depth value in mm (no conversion)
        float z_mm = fabsf(positions[i].z);

        // s_x and s_y: lateral uncertainty in mm
        // simplifies to z_mm / fx_depth (in mm)
        float s_x = z_mm / fx_depth;
        float s_y = z_mm / fy_depth;
        // s_z: depth uncertainty in mm
        float s_z = (depth_noise_a * z_mm - depth_noise_b);

        // Variance in mm^2
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
        // Use mm consistently
        float d = static_cast<float>(depth.at<uint16_t>(y, x));
        return (d <= 0.0f || d > 10000.0f) ? -1.0f : d;
        };

    float dz = getDepth(u, v);
    float dz_du = getDepth(u + 1, v);
    float dz_dv = getDepth(u, v + 1);

    if (dz <= 0.0f || dz_du <= 0.0f || dz_dv <= 0.0f)
        return glm::vec3(0, 0, 0);   // invalid normal

    // Convert pixels to camera coords (mm)
    glm::vec3 p((u - cx) * dz / fx, (v - cy) * dz / fy, dz);
    glm::vec3 px((u + 1 - cx) * dz_du / fx, (v - cy) * dz_du / fy, dz_du);
    glm::vec3 py((u - cx) * dz_dv / fx, (v + 1 - cy) * dz_dv / fy, dz_dv);

    glm::vec3 vx = px - p;
    glm::vec3 vy = py - p;

    glm::vec3 n = glm::normalize(glm::cross(vx, vy));

    // Ensure normal faces the camera (positive Z)
    //if (n.z < 0) n = -n;

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
    //dst.scales.storeData(dst.scales_cpu.data(), dst.num.gaussians, 4*sizeof(float), 0, useCudaGLInterop, false, true);

    //dst.rotations_cpu = std::vector<glm::vec4>(dst.num_gaussians);
    //reader.extract_properties(rot_idx, 4, miniply::PLYPropertyType::Float, dst.rotations_cpu.data());
    //dst.rotations.storeData(dst.rotations_cpu.data(), dst.num.gaussians, 4*sizeof(float), 0, useCudaGLInterop, false, true);

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
        dst.opacities_cpu[i] = sigmoid(exp(-sdf_val * sdf_val / 0.01f)); // apply sigmoid activation sdf[i] = h_SDF_Torus(pts[i], 0.6f, 0.4f);
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

    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2*sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4*sizeof(float), 0, useCudaGLInterop, true, true);

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

    // Extract camera intrinsics (in pixels)
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
            // Get depth value in millimeters
            float depth_mm = static_cast<float>(depth_image.at<uint16_t>(i, j));

            // Skip invalid depth values (keep same max range, in mm)
            if (depth_mm <= 0.0f || depth_mm > 2000.0f) continue;

            // Convert pixel coordinates to 3D point in depth camera coordinates (mm)
            float x_d_mm = (j - CX_DEPTH) * depth_mm / FX_DEPTH;
            float y_d_mm = (i - CY_DEPTH) * depth_mm / FY_DEPTH;
            float z_d_mm = depth_mm;

            // Transform to RGB camera coordinates using R and T (all in mm)
            glm::vec3 point_depth_mm = glm::vec3(x_d_mm, y_d_mm, z_d_mm);
            glm::vec3 point_rgb_mm = R * point_depth_mm + T;

            float x_rgb_mm = point_rgb_mm.x;
            float y_rgb_mm = point_rgb_mm.y;
            float z_rgb_mm = point_rgb_mm.z;

            // Skip points behind the RGB camera
            if (z_rgb_mm <= 0.0f) continue;

            // Project to RGB image coordinates (units cancel)
            float u = (x_rgb_mm * FX_RGB) / z_rgb_mm + CX_RGB;
            float v = (y_rgb_mm * FY_RGB) / z_rgb_mm + CY_RGB;

            // Check if projection is within RGB image bounds
            if (u < 0 || u >= w_rgb || v < 0 || v >= h_rgb) continue;

            // Clip to RGB image bounds
            int u_int = std::max(0, std::min(w_rgb - 1, (int)round(u)));
            int v_int = std::max(0, std::min(h_rgb - 1, (int)round(v)));

            // Get RGB color values (BGR format, normalize to [0,1])
            cv::Vec3b color = rgb_image.at<cv::Vec3b>(v_int, u_int);

            // Transform to world coordinates using rotation and translation (mm)
            glm::vec3 pos_world_mm = rgbToWorldR * (point_rgb_mm + rgbToWorldT);

            // Store output positions in mm
            glm::vec4 pos_world_h = glm::vec4(pos_world_mm, 1.0f);

            // Compute normals from depth (mm)
            glm::vec3 normal_cam = computeDepthNormal(
                depth_image, j, i,
                FX_DEPTH, FY_DEPTH,
                CX_DEPTH, CY_DEPTH
            );

            if (normal_cam == glm::vec3(0) || std::isnan(normal_cam.x) || std::isnan(normal_cam.y) || std::isnan(normal_cam.z)) continue; // skip invalid normals

            glm::vec3 normal_world = convertNormalToWorld(
                normal_cam,
                R,                // depth Å® RGB rotation
                rgbToWorldR       // RGB Å® world rotation
            );
            glm::vec4 normal_world_h = glm::vec4(normal_world, 0.0f);

            temp_positions.push_back(pos_world_h);
            temp_normals.push_back(normal_world_h);
            temp_colors.push_back(glm::vec3(color[2] / 255.0f, color[1] / 255.0f, color[0] / 255.0f));
            depth_cam_points.push_back(point_depth_mm); // Store depth camera points (mm) for covariance calculation
        }
    }

    dst.num_gaussians = (int)temp_positions.size();

    if (dst.num_gaussians == 0) {
        std::cerr << "Error: No valid points generated from RGBD images!" << std::endl;
        return;
    }

    std::cout << "Generated " << dst.num_gaussians << " points from RGBD images" << std::endl;

    // Store positions in open gl convention
    dst.positions_cpu = temp_positions;
    for (int i = 0; i < dst.num_gaussians; i++) {
        dst.positions_cpu[i].y = -dst.positions_cpu[i].y;
        dst.positions_cpu[i].z = -dst.positions_cpu[i].z;
    }
    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Store normals
    dst.normals_cpu = temp_normals;
    for (int i = 0; i < dst.num_gaussians; i++) {
        dst.normals_cpu[i].y = -dst.normals_cpu[i].y;
        dst.normals_cpu[i].z = -dst.normals_cpu[i].z;
    }
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
        glm::mat3 C_rgb = R * C * glm::transpose(R);
        glm::mat3 C_world = rgbToWorldR * C_rgb * glm::transpose(rgbToWorldR); // to world coords
        dst.covX_cpu[k] = glm::vec4(C_world[0][0], C_world[0][1], C_world[0][2], 0.0f);
        dst.covY_cpu[k] = glm::vec4(C_world[1][0], C_world[1][1], C_world[1][2], 0.0f);
        dst.covZ_cpu[k] = glm::vec4(C_world[2][0], C_world[2][1], C_world[2][2], 0.0f);
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
    const int total = nA + nB;
    
    // Check available GPU memory before proceeding
    size_t free_mem, total_mem;
    cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_err != cudaSuccess) {
        std::cerr << "Failed to query GPU memory: " << cudaGetErrorString(mem_err) << std::endl;
        dst.initialized = false;
        return;
    }
    
    // Estimate required memory (rough calculation)
    // Each gaussian needs: position(16B) + opacity(4B) + normals(16B) + covariance(48B) + SH(192B) + misc(~100B) ? 400 bytes
    size_t estimated_mem = total * 400;
    
    std::cout << "Merge: GPU Memory - Free: " << (free_mem / (1024.0*1024.0)) 
              << " MB, Estimated need: " << (estimated_mem / (1024.0*1024.0)) << " MB" << std::endl;
    
    if (free_mem < estimated_mem * 1.5) { // Need 1.5x for safety margin
        std::cerr << "WARNING: Low GPU memory for merge. Attempting anyway..." << std::endl;
    }
    
    dst.num_gaussians = total;

    // --- HELPER LAMBDAS ---

    // 1. Read vec4 from GPU
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

    // 2. Read float from GPU
    auto readFloatFromGLBuffer = [](const GLBuffer& buf, int count)->std::vector<float> {
        if (count == 0) return {};
        return const_cast<GLBuffer&>(buf).getAsFloats(count);
        };

    // 3. Robust Fetcher: Tries CPU first, then GPU, then returns empty
    auto fetchVec4 = [&](const GaussianCloud& g, const std::vector<glm::vec4>& cpu_vec, const GLBuffer& gpu_buf, int count) {
        if (!cpu_vec.empty()) return cpu_vec;
        if (g.initialized && count > 0) return readVec4FromGLBuffer(gpu_buf, count);
        return std::vector<glm::vec4>();
        };

    auto fetchFloat = [&](const GaussianCloud& g, const std::vector<float>& cpu_vec, const GLBuffer& gpu_buf, int count) {
        if (!cpu_vec.empty()) return cpu_vec;
        if (g.initialized && count > 0) return readFloatFromGLBuffer(gpu_buf, count);
        return std::vector<float>();
        };

    // --- STEP 1: SNAPSHOT DATA (Fixes Aliasing Bug) ---
    // We read all source data into local variables first. 
    // This allows dst to be one of the inputs (e.g., merge(A, A, B)) without data loss.

    // Positions
    std::vector<glm::vec4> posA, posB;
    if (hasA) posA = fetchVec4(a, a.positions_cpu, a.positions, nA);
    if (hasB) posB = fetchVec4(b, b.positions_cpu, b.positions, nB);
    
    // Normals
    std::vector<glm::vec4> normA, normB;
    if (hasA) normA = fetchVec4(a, a.normals_cpu, a.normals, nA);
    if (hasB) normB = fetchVec4(b, b.normals_cpu, b.normals, nB);
    
    // Opacities
    std::vector<float> opacA, opacB;
    if (hasA) opacA = fetchFloat(a, a.opacities_cpu, a.opacities, nA);
    if (hasB) opacB = fetchFloat(b, b.opacities_cpu, b.opacities, nB);

    // SDF
    std::vector<float> sdfA, sdfB;
    if (hasA) sdfA = fetchFloat(a, a.sdf_cpu, a.sdf, nA);
    if (hasB) sdfB = fetchFloat(b, b.sdf_cpu, b.sdf, nB);

    // Covariances (X, Y, Z rows)
    std::vector<glm::vec4> covXA, covYA, covZA;
    std::vector<glm::vec4> covXB, covYB, covZB;

    if (hasA) {
        covXA = fetchVec4(a, a.covX_cpu, a.covariance[0], nA);
        covYA = fetchVec4(a, a.covY_cpu, a.covariance[1], nA);
        covZA = fetchVec4(a, a.covZ_cpu, a.covariance[2], nA);
    }
    if (hasB) {
        covXB = fetchVec4(b, b.covX_cpu, b.covariance[0], nB);
        covYB = fetchVec4(b, b.covY_cpu, b.covariance[1], nB);
        covZB = fetchVec4(b, b.covZ_cpu, b.covariance[2], nB);
    }

    // SH Coefficients (Special case: always float arrays)
    // We need 3 channels (R, G, B)
    std::vector<float> shA[3], shB[3];
    for (int ch = 0; ch < 3; ch++) {
        if (hasA) shA[ch] = readFloatFromGLBuffer(a.sh_coeffs[ch], nA * 16);
        if (hasB) shB[ch] = readFloatFromGLBuffer(b.sh_coeffs[ch], nB * 16);
    }

    // --- STEP 2: CLEAR AND REBUILD DESTINATION ---

    // 2.1 Positions
    dst.positions_cpu.clear();
    dst.positions_cpu.reserve(dst.num_gaussians);
    if (hasA && !posA.empty()) dst.positions_cpu.insert(dst.positions_cpu.end(), posA.begin(), posA.end());
    else if (hasA) dst.positions_cpu.insert(dst.positions_cpu.end(), nA, glm::vec4(0.f)); // Fallback safety

    if (hasB && !posB.empty()) dst.positions_cpu.insert(dst.positions_cpu.end(), posB.begin(), posB.end());
    else if (hasB) dst.positions_cpu.insert(dst.positions_cpu.end(), nB, glm::vec4(0.f)); // Fallback safety

    dst.positions.storeData(dst.positions_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    
    // 2.5 Normals
    dst.normals_cpu.clear();
    dst.normals_cpu.reserve(dst.num_gaussians);

    if (hasA && !normA.empty()) dst.normals_cpu.insert(dst.normals_cpu.end(), normA.begin(), normA.end());
    else if (hasA) dst.normals_cpu.insert(dst.normals_cpu.end(), nA, glm::vec4(0, 0, 1, 0));

    if (hasB && !normB.empty()) dst.normals_cpu.insert(dst.normals_cpu.end(), normB.begin(), normB.end());
    else if (hasB) dst.normals_cpu.insert(dst.normals_cpu.end(), nB, glm::vec4(0, 0, 1, 0));

    dst.normals.storeData(dst.normals_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    
    // 2.2 Opacities
    dst.opacities_cpu.clear();
    dst.opacities_cpu.reserve(dst.num_gaussians);
    if (hasA && !opacA.empty()) dst.opacities_cpu.insert(dst.opacities_cpu.end(), opacA.begin(), opacA.end());
    else if (hasA) dst.opacities_cpu.insert(dst.opacities_cpu.end(), nA, 1.0f); // Default 1.0

    if (hasB && !opacB.empty()) dst.opacities_cpu.insert(dst.opacities_cpu.end(), opacB.begin(), opacB.end());
    else if (hasB) dst.opacities_cpu.insert(dst.opacities_cpu.end(), nB, 1.0f); // Default 1.0

    dst.opacities.storeData(dst.opacities_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

    // 2.3 SDF
    dst.sdf_cpu.clear();
    dst.sdf_cpu.reserve(dst.num_gaussians);
    if (hasA && !sdfA.empty()) dst.sdf_cpu.insert(dst.sdf_cpu.end(), sdfA.begin(), sdfA.end());
    else if (hasA) dst.sdf_cpu.insert(dst.sdf_cpu.end(), nA, 0.0f);

    if (hasB && !sdfB.empty()) dst.sdf_cpu.insert(dst.sdf_cpu.end(), sdfB.begin(), sdfB.end());
    else if (hasB) dst.sdf_cpu.insert(dst.sdf_cpu.end(), nB, 0.0f);

    dst.sdf.storeData(dst.sdf_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

    // 2.4 SH Coeffs
    for (int ch = 0; ch < 3; ++ch) {
        std::vector<float> merged(16 * dst.num_gaussians);
        size_t offset = 0;

        // A
        if (hasA) {
            if (!shA[ch].empty()) {
                std::memcpy(merged.data() + offset, shA[ch].data(), shA[ch].size() * sizeof(float));
            }
            else {
                // Fallback: DC=0.5, others 0
                for (int i = 0; i < nA; i++) {
                    merged[offset + i * 16 + 0] = 0.5f;
                    for (int l = 1; l < 16; l++) merged[offset + i * 16 + l] = 0.0f;
                }
            }
            offset += 16 * nA;
        }
        // B
        if (hasB) {
            if (!shB[ch].empty()) {
                std::memcpy(merged.data() + offset, shB[ch].data(), shB[ch].size() * sizeof(float));
            }
            else {
                // Fallback
                for (int i = 0; i < nB; i++) {
                    merged[offset + i * 16 + 0] = 0.5f;
                    for (int l = 1; l < 16; l++) merged[offset + i * 16 + l] = 0.0f;
                }
            }
            offset += 16 * nB;
        }
        dst.sh_coeffs[ch].storeData(merged.data(), dst.num_gaussians, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
    }

    // 2.6 Covariance
    dst.covX_cpu.clear(); dst.covY_cpu.clear(); dst.covZ_cpu.clear();
    dst.covX_cpu.reserve(dst.num_gaussians);
    dst.covY_cpu.reserve(dst.num_gaussians);
    dst.covZ_cpu.reserve(dst.num_gaussians);

    // Helper to insert with fallback
    auto insertCov = [&](std::vector<glm::vec4>& target, const std::vector<glm::vec4>& source, int count) {
        if (!source.empty()) target.insert(target.end(), source.begin(), source.end());
        else target.insert(target.end(), count, glm::vec4(0.01f, 0, 0, 0)); // Dummy fallback
        };

    if (hasA) {
        insertCov(dst.covX_cpu, covXA, nA);
        insertCov(dst.covY_cpu, covYA, nA);
        insertCov(dst.covZ_cpu, covZA, nA);
    }
    if (hasB) {
        insertCov(dst.covX_cpu, covXB, nB);
        insertCov(dst.covY_cpu, covYB, nB);
        insertCov(dst.covZ_cpu, covZB, nB);
    }

    // Ensure size consistency / Last-ditch fallback
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

    // Upload Covariance to GPU
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

    // --- STEP 3: DERIVED BUFFERS ---

    dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
    dst.gaussians_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.gaussians_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);
    dst.sorted_depths.storeData(nullptr, dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, true, true);
    dst.sorted_gaussian_indices.storeData(nullptr, dst.num_gaussians, sizeof(int), 0, useCudaGLInterop, true, true);

    dst.bounding_boxes.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.conic_opacity.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.eigen_vecs.storeData(nullptr, dst.num_gaussians, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
    dst.predicted_colors.storeData(nullptr, dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, true, true);

    dst.initialized = true;

	// write to txt file for debugging
    // write to txt file for debugging
    std::ofstream debug_file("merged_gaussian_cloud_debug.txt");
    if (debug_file.is_open()) {
        debug_file << "Merged Gaussian Cloud Debug Output\n";
        debug_file << "===================================\n";
        debug_file << "Total Gaussians: " << dst.num_gaussians << "\n";
        debug_file << "Source A: " << nA << " gaussians\n";
        debug_file << "Source B: " << nB << " gaussians\n";
        debug_file << "===================================\n\n";

        // Write header
        debug_file << "Index,PosX,PosY,PosZ,NormalX,NormalY,NormalZ,Opacity,SDF,"
            << "CovXX,CovXY,CovXZ,CovYX,CovYY,CovYZ,CovZX,CovZY,CovZZ,"
            << "SH_R_DC,SH_G_DC,SH_B_DC\n";

        // Write data for each Gaussian
        for (int i = 0; i < dst.num_gaussians; i++) {
            // Position
            debug_file << i << ","
                << dst.positions_cpu[i].x << "," << dst.positions_cpu[i].y << "," << dst.positions_cpu[i].z << ",";

            // Normal
            if (i < (int)dst.normals_cpu.size()) {
                debug_file << dst.normals_cpu[i].x << "," << dst.normals_cpu[i].y << "," << dst.normals_cpu[i].z << ",";
            }
            else {
                debug_file << "0,0,1,";
            }

            // Opacity
            if (i < (int)dst.opacities_cpu.size()) {
                debug_file << dst.opacities_cpu[i] << ",";
            }
            else {
                debug_file << "1.0,";
            }

            // SDF
            if (i < (int)dst.sdf_cpu.size()) {
                debug_file << dst.sdf_cpu[i] << ",";
            }
            else {
                debug_file << "0.0,";
            }

            // Covariance matrix (3x3)
            if (i < (int)dst.covX_cpu.size() && i < (int)dst.covY_cpu.size() && i < (int)dst.covZ_cpu.size()) {
                debug_file << dst.covX_cpu[i].x << "," << dst.covX_cpu[i].y << "," << dst.covX_cpu[i].z << ","
                    << dst.covY_cpu[i].x << "," << dst.covY_cpu[i].y << "," << dst.covY_cpu[i].z << ","
                    << dst.covZ_cpu[i].x << "," << dst.covZ_cpu[i].y << "," << dst.covZ_cpu[i].z << ",";
            }
            else {
                debug_file << "0.01,0,0,0,0.01,0,0,0,0.01,";
            }

            // SH coefficients (DC component only for R,G,B)
            debug_file << "SH_data_in_buffer,SH_data_in_buffer,SH_data_in_buffer";

            debug_file << "\n";
        }

        debug_file << "\n===================================\n";
        debug_file << "Note: SH coefficients are stored in GPU buffers (16 coeffs x 3 channels)\n";
        debug_file << "To inspect SH data, it needs to be read back from dst.sh_coeffs[0-2]\n";
        debug_file.close();

        std::cout << "Debug: Merged Gaussian cloud written to 'merged_gaussian_cloud_debug.txt'\n";
    }
    else {
        std::cerr << "Error: Could not open debug file for writing!\n";
    }

    std::cout << "Merged clouds: " << nA << " + " << nB
        << " = " << dst.num_gaussians << " gaussians.\n";
}