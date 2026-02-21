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
#include "RgbdLoadCuda.cuh"
#include <filesystem>
#include <algorithm>
#include <map>
#include <iomanip>
#include <sstream>

using namespace glm;

// Helper: convert glm::mat3 (column-major) to a float[9] row-major array for CUDA kernels
static void mat3ToRowMajor(const glm::mat3& m, float out[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r * 3 + c] = m[c][r]; // GLM is column-major
}

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

struct SurfaceBasis {
    glm::vec3 normal; // normal
    glm::vec3 tangent; // tangent x
    glm::vec3 bitangent; // tangent y
};

// Helper function to calculate depth-dependent covariance (all units in mm)
static void calculateDepthCovariance(const std::vector<glm::vec3>& positions,
    float fx_depth, float fy_depth, std::vector<glm::vec3> tangents,   
    std::vector<glm::mat3>& covariances) {
    // Noise model parameters adjusted for mm units
    // depth_noise_a and depth_noise_b are relative to depth in mm
    const float depth_noise_a = 0.0007f;   // dimensionless coefficient
    const float depth_noise_b = 0.2f;      // in mm (was 0.0002m = 0.2mm)
    const float scale_factor = 1.0f;
    const float pix_size = 0.0035f;

    covariances.resize(positions.size());

    for (size_t i = 0; i < positions.size(); i++) {
        // Use absolute depth value in mm (no conversion)
        float z_mm = fabsf(positions[i].z);

        // s_x and s_y: lateral uncertainty in mm
        // simplifies to z_mm / fx_depth (in mm)
        float s_x = z_mm / fx_depth;
        float s_y = z_mm / fy_depth;
        float z_noise = (depth_noise_a * z_mm - depth_noise_b); // s_z: depth uncertainty in mm
        float dzx = fabs(s_x * tangents[i].x);
        float dzy = fabs(s_y * tangents[i].y);
        float s_z = (dzx + dzy) + z_noise;

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
 
SurfaceBasis computeDepthBasis(
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

    if (dz <= 0.0f || dz_du <= 0.0f || dz_dv <= 0.0f) {
        return { glm::vec3(0), glm::vec3(0), glm::vec3(0) };
    }

    // Convert pixels to camera coords (mm)
    glm::vec3 p((u - cx) * dz / fx, (v - cy) * dz / fy, dz);
    glm::vec3 px((u + 1 - cx) * dz_du / fx, (v - cy) * dz_du / fy, dz_du);
    glm::vec3 py((u - cx) * dz_dv / fx, (v + 1 - cy) * dz_dv / fy, dz_dv);

    glm::vec3 vx = px - p;
    glm::vec3 vy = py - p;

    glm::vec3 n = glm::normalize(glm::cross(vx, vy));
    glm::vec3 t = vx;
    glm::vec3 b = vy;
    // Ensure normal faces the camera (positive Z)
    if (n.z < 0.0f) {
        n = -n;
        t = -t;
    }

    return { n, t, b };
}

static inline float choose_scale_neus(float depth_mm, const glm::vec3& n_cam, const glm::vec3& p_cam_mm)
{
    // --- slide parameters ---
    const float D = 1500.0f;   // max distance (mm) used for normalization
    const float S_near = 1000000.0f;   // near -> larger s (sharper)
    const float S_far = 100000.0f;      // far  -> smaller s (blurrier)
    const float alpha = 1.0f;      // orientation strength (your slide's α)

    // ---------------- distance effect: (S_near - (S_near - S_far) * (d/D)^2) ----------------
    float x = depth_mm / D;
    x = std::clamp(x, 0.0f, 1.0f);

    float s_dist = S_near - (S_near - S_far) * (x * x); // near high, far low

    // ---------------- orientation effect: (1 + α / |n·v|) ----------------
    glm::vec3 v = glm::normalize(-p_cam_mm);  // point -> camera (camera space)
    float cos_nv = std::abs(glm::dot(glm::normalize(n_cam), v));
    cos_nv = std::clamp(cos_nv, 0.05f, 1.0f); // avoid blow-up at grazing

    float orient = 1.0f + alpha * (1.0f / cos_nv);
    orient = std::clamp(orient, 1.0f, 6.0f);  // cap for stability

    float s = s_dist * orient;
    return std::clamp(s, S_far, S_near);
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


void PointCloudLoader::loadRdm(GaussianCloud& dst, int nb_pts, bool useCudaGLInterop) {
    dst.freeRawCudaBuffers();
    dst.clearCpuData();
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

    dst.normals_cpu = std::vector<glm::vec4>(dst.num_gaussians, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    dst.normals.storeData(dst.normals_cpu.data(), dst.num_gaussians, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    dst.scale_neus_cpu = std::vector<float>(dst.num_gaussians, 500000.0f);
    dst.scale_neus.storeData(dst.scale_neus_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

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
	std::vector<glm::vec3> temp_tangents_cam;
    std::vector<float> temp_opacities;
    std::vector<float> temp_scale_neus;

    for (int i = 0; i < h_d; i++) {
        for (int j = 0; j < w_d; j++) {
            // Get depth value in millimeters
            float depth_mm = static_cast<float>(depth_image.at<uint16_t>(i, j));

            // Skip invalid depth values (keep same max range, in mm)
            if (depth_mm <= 0.0f || depth_mm > 1800.0f) continue;

            // Convert pixel coordinates to 3D point in depth camera coordinates (mm)
            float x_d_mm = (j - CX_DEPTH) * depth_mm / FX_DEPTH;
            float y_d_mm = (i - CY_DEPTH) * depth_mm / FY_DEPTH;
            float z_d_mm = depth_mm;

            // Transform to RGB camera coordinates using R and T (in mm)
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

			// Remove 100 pixels border to avoid invalid colors
			//if (u < 400 || u >= w_rgb - 400 || v < 0 || v >= h_rgb - 100) continue;

            // Clip to RGB image bounds
            int u_int = std::max(0, std::min(w_rgb - 1, (int)round(u)));
            int v_int = std::max(0, std::min(h_rgb - 1, (int)round(v)));

            // Get RGB color values (BGR format, normalize to [0,1])
            cv::Vec3b color = rgb_image.at<cv::Vec3b>(v_int, u_int);

            // Transform to world coordinates using rotation and translation (mm)
            glm::vec3 pos_world_mm = rgbToWorldR * point_rgb_mm + rgbToWorldT;

            // Store output positions in mm
            glm::vec4 pos_world_h = glm::vec4(pos_world_mm, 1.0f);

            float opacity;
            // Compute opacities (1.0f until 1m then decay to 0 at 2m)
            if (depth_mm <= 1000.0f) {
                opacity = 1.0f;
            }
            else {
                float lambda = -log(0.01f) / (2.0f - 1.0f); // Decay rate
                opacity = exp(-lambda * (depth_mm - 1000.0f));
            }

            // Compute normals from depth (mm)
            SurfaceBasis basis = computeDepthBasis(
                depth_image, j, i,
                FX_DEPTH, FY_DEPTH,
                CX_DEPTH, CY_DEPTH
            );
            glm::vec3 t = basis.tangent;
            glm::vec3 n = basis.normal;

            if (n == glm::vec3(0) || std::isnan(n.x) || std::isnan(n.y) || std::isnan(n.z)) continue; // skip invalid normals

            glm::vec3 normal_world = convertNormalToWorld(
                n,
                R,                // depth → RGB rotation
                rgbToWorldR       // RGB → world rotation
            );
            glm::vec4 normal_world_h = glm::vec4(normal_world, 0.0f);

			// Choose scale for NeuS representation
            float scale_neus = choose_scale_neus(depth_mm, n, point_depth_mm);
            temp_scale_neus.push_back(scale_neus);

            temp_positions.push_back(pos_world_h);
            temp_opacities.push_back(opacity);
            temp_normals.push_back(normal_world_h);
			temp_tangents_cam.push_back(glm::vec3(t));
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
    // dst.opacities_cpu = std::vector<float>(dst.num_gaussians, 1.0f);
    dst.opacities_cpu = temp_opacities;
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
    calculateDepthCovariance(depth_cam_points, FX_DEPTH, FY_DEPTH, temp_tangents_cam, covariance_matrices);

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

    // Store scale for NeuS representation (after the normals upload)
    dst.scale_neus_cpu = temp_scale_neus;
    dst.scale_neus.storeData(dst.scale_neus_cpu.data(), dst.num_gaussians, sizeof(float), 0, useCudaGLInterop, false, true);

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

    dst.initialized = true;

    std::cout << "Finished loading RGBD point cloud." << std::endl;
}


void PointCloudLoader::merge(GaussianCloud& dst,
    const GaussianCloud& a,
    const GaussianCloud& b,
    bool useCudaGLInterop)
{
    const bool hasA = a.initialized && a.num_gaussians > 0;
    const bool hasB = b.initialized && b.num_gaussians > 0;

    if (!hasA && !hasB) {
        dst.initialized = false;
        dst.num_gaussians = 0;
        return;
    }

    const int nA = hasA ? a.num_gaussians : 0;
    const int nB = hasB ? b.num_gaussians : 0;
    const int total = nA + nB;

    const bool sameSize = dst.initialized && dst.num_gaussians == total;
    dst.num_gaussians = total;

    // --- HELPER: GPU-side copy from source GLBuffer into dst GLBuffer at byte offset ---
    auto gpuCopyRegion = [](const GLBuffer& src, GLBuffer& dst_buf, size_t srcBytes, size_t dstByteOffset) {
        if (srcBytes == 0) return;
        void* srcPtr = const_cast<GLBuffer&>(src).getCudaPtr();
        void* dstPtr = dst_buf.getCudaPtr();
        if (srcPtr && dstPtr) {
            cudaMemcpy(static_cast<char*>(dstPtr) + dstByteOffset, srcPtr, srcBytes, cudaMemcpyDeviceToDevice);
        }
        };

    // --- HELPER: upload vec4 buffer by GPU-concatenation (no CPU round-trip) ---
    auto mergeVec4Buffer = [&](GLBuffer& dst_buf, const GLBuffer& srcA, int countA,
        const GLBuffer& srcB, int countB,
        std::vector<glm::vec4>& cpu_vec) {
            // Ensure the GL buffer exists at the right size
            dst_buf.storeOrUpdateData(nullptr, total, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
            if (hasA) gpuCopyRegion(srcA, dst_buf, countA * 4 * sizeof(float), 0);
            if (hasB) gpuCopyRegion(srcB, dst_buf, countB * 4 * sizeof(float), countA * 4 * sizeof(float));
            // Keep CPU mirror in sync (needed for GUI debug display)
            cpu_vec.resize(total);
            if (dst_buf.getCudaPtr())
                cudaMemcpy(cpu_vec.data(), dst_buf.getCudaPtr(), total * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        };

    // --- HELPER: upload float buffer by GPU-concatenation ---
    auto mergeFloatBuffer = [&](GLBuffer& dst_buf, const GLBuffer& srcA, int countA,
        const GLBuffer& srcB, int countB,
        std::vector<float>& cpu_vec) {
            dst_buf.storeOrUpdateData(nullptr, total, sizeof(float), 0, useCudaGLInterop, false, true);
            if (hasA) gpuCopyRegion(srcA, dst_buf, countA * sizeof(float), 0);
            if (hasB) gpuCopyRegion(srcB, dst_buf, countB * sizeof(float), countA * sizeof(float));
            cpu_vec.resize(total);
            if (dst_buf.getCudaPtr())
                cudaMemcpy(cpu_vec.data(), dst_buf.getCudaPtr(), total * sizeof(float), cudaMemcpyDeviceToHost);
        };

    // --- HELPER: upload SH buffer (16 floats per element) by GPU-concatenation ---
    auto mergeSHBuffer = [&](GLBuffer& dst_buf, const GLBuffer& srcA, int countA,
        const GLBuffer& srcB, int countB) {
            dst_buf.storeOrUpdateData(nullptr, total, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
            if (hasA) gpuCopyRegion(srcA, dst_buf, countA * 16 * sizeof(float), 0);
            if (hasB) gpuCopyRegion(srcB, dst_buf, countB * 16 * sizeof(float), countA * 16 * sizeof(float));
        };

    // --- Merge data buffers (GPU-to-GPU, no CPU readback) ---
    // Use dummy empty GLBuffer for the missing source when only one cloud is present
    static GLBuffer s_empty;

    const GLBuffer& posA = hasA ? a.positions : s_empty;
    const GLBuffer& posB = hasB ? b.positions : s_empty;
    mergeVec4Buffer(dst.positions, posA, nA, posB, nB, dst.positions_cpu);

    const GLBuffer& normA = hasA ? a.normals : s_empty;
    const GLBuffer& normB = hasB ? b.normals : s_empty;
    mergeVec4Buffer(dst.normals, normA, nA, normB, nB, dst.normals_cpu);

    mergeFloatBuffer(dst.opacities, hasA ? a.opacities : s_empty, nA,
        hasB ? b.opacities : s_empty, nB, dst.opacities_cpu);

    mergeFloatBuffer(dst.sdf, hasA ? a.sdf : s_empty, nA,
        hasB ? b.sdf : s_empty, nB, dst.sdf_cpu);

    for (int ch = 0; ch < 3; ch++) {
        mergeSHBuffer(dst.sh_coeffs[ch],
            hasA ? a.sh_coeffs[ch] : s_empty, nA,
            hasB ? b.sh_coeffs[ch] : s_empty, nB);
    }

    for (int row = 0; row < 3; row++) {
        const GLBuffer& covA = hasA ? a.covariance[row] : s_empty;
        const GLBuffer& covB = hasB ? b.covariance[row] : s_empty;
        std::vector<glm::vec4>& cpu = (row == 0 ? dst.covX_cpu : (row == 1 ? dst.covY_cpu : dst.covZ_cpu));
        mergeVec4Buffer(dst.covariance[row], covA, nA, covB, nB, cpu);
    }

    // Merge per-gaussian scale_neus buffer
    mergeFloatBuffer(dst.scale_neus, hasA ? a.scale_neus : s_empty, nA,
        hasB ? b.scale_neus : s_empty, nB, dst.scale_neus_cpu);

    // --- Derived / scratch buffers: only reallocate when total changes ---
    if (!sameSize) {
        dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
        dst.gaussians_depths.storeData(nullptr, total, sizeof(float), 0, useCudaGLInterop, true, true);
        dst.gaussians_indices.storeData(nullptr, total, sizeof(int), 0, useCudaGLInterop, true, true);
        dst.sorted_depths.storeData(nullptr, total, sizeof(float), 0, useCudaGLInterop, true, true);
        dst.sorted_gaussian_indices.storeData(nullptr, total, sizeof(int), 0, useCudaGLInterop, true, true);

        dst.bounding_boxes.storeData(nullptr, total, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.conic_opacity.storeData(nullptr, total, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.eigen_vecs.storeData(nullptr, total, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.predicted_colors.storeData(nullptr, total, 4 * sizeof(float), 0, useCudaGLInterop, true, true);

        // Free and reallocate raw CUDA buffers
        dst.freeRawCudaBuffers();
    }

    dst.initialized = true;

    std::cout << "Merged clouds: " << nA << " + " << nB
        << " = " << dst.num_gaussians << " gaussians.\n";
}

RgbdFrameSequence PointCloudLoader::discoverFrameSequence(
    const std::string& depthDir,
    const std::string& colorDir,
    const std::string& prefix,
    const std::string& extension,
    int numDigits)
{
    RgbdFrameSequence seq;

    if (!std::filesystem::exists(depthDir) || !std::filesystem::exists(colorDir)) {
        std::cerr << "discoverFrameSequence: directory does not exist. depth=" 
                  << depthDir << " color=" << colorDir << std::endl;
        return seq;
    }

    // Collect all matching depth files
    std::map<int, std::string> depthFiles;
    for (const auto& entry : std::filesystem::directory_iterator(depthDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        
        // Check prefix and extension
        if (filename.size() < prefix.size() + numDigits + extension.size()) continue;
        if (filename.substr(0, prefix.size()) != prefix) continue;
        if (filename.substr(filename.size() - extension.size()) != extension) continue;
        
        // Extract frame number
        std::string numStr = filename.substr(prefix.size(), filename.size() - prefix.size() - extension.size());
        try {
            int frameNum = std::stoi(numStr);
            depthFiles[frameNum] = entry.path().string();
        } catch (...) {
            continue;
        }
    }

    // Collect all matching color files
    std::map<int, std::string> colorFiles;
    for (const auto& entry : std::filesystem::directory_iterator(colorDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        
        if (filename.size() < prefix.size() + numDigits + extension.size()) continue;
        if (filename.substr(0, prefix.size()) != prefix) continue;
        if (filename.substr(filename.size() - extension.size()) != extension) continue;
        
        std::string numStr = filename.substr(prefix.size(), filename.size() - prefix.size() - extension.size());
        try {
            int frameNum = std::stoi(numStr);
            colorFiles[frameNum] = entry.path().string();
        } catch (...) {
            continue;
        }
    }

    // Find common frames (present in both depth and color)
    for (const auto& [frameNum, depthPath] : depthFiles) {
        auto it = colorFiles.find(frameNum);
        if (it != colorFiles.end()) {
            seq.depthPaths.push_back(depthPath);
            seq.colorPaths.push_back(it->second);
        }
    }

    seq.totalFrames = (int)seq.depthPaths.size();
    seq.currentFrame = 0;

    std::cout << "discoverFrameSequence: found " << seq.totalFrames 
              << " matching frames in " << depthDir << std::endl;

    return seq;
}

std::future<PreloadedFrame> PointCloudLoader::preloadFrameAsync(
    const std::string& depthPath, const std::string& colorPath)
{
    return std::async(std::launch::async, [depthPath, colorPath]() -> PreloadedFrame {
        PreloadedFrame f;
        auto depth = std::make_shared<cv::Mat>(cv::imread(depthPath, cv::IMREAD_ANYDEPTH));
        auto color = std::make_shared<cv::Mat>(cv::imread(colorPath, cv::IMREAD_COLOR));
        f.valid = !depth->empty() && !color->empty();
        f.depth = depth;
        f.color = color;
        return f;
    });
}

void PointCloudLoader::loadRgbdGpuFromMats(GaussianCloud& dst,
    const PreloadedFrame& frame,
    const glm::mat3& depth_intrinsics,
    const glm::mat3& rgb_intrinsics,
    const glm::mat3& R,
    const glm::vec3& T,
    const glm::mat3& rgbToWorldR,
    const glm::vec3& rgbToWorldT,
    bool useCudaGLInterop)
{
    if (!frame.valid || !frame.depth || !frame.color) {
        std::cerr << "Error: Invalid PreloadedFrame passed to loadRgbdGpuFromMats!" << std::endl;
        return;
    }
    loadRgbdGpuInternal(dst, frame.depth.get(), frame.color.get(),
        depth_intrinsics, rgb_intrinsics, R, T, rgbToWorldR, rgbToWorldT, useCudaGLInterop);
}

void PointCloudLoader::loadRgbdGpu(GaussianCloud& dst, const std::string& depth_path,
    const std::string& rgb_path,
    const glm::mat3& depth_intrinsics,
    const glm::mat3& rgb_intrinsics,
    const glm::mat3& R,
    const glm::vec3& T,
    const glm::mat3& rgbToWorldR,
    const glm::vec3& rgbToWorldT,
    bool useCudaGLInterop)
{
    cv::Mat depth_image = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
    cv::Mat rgb_image = cv::imread(rgb_path, cv::IMREAD_COLOR);

    loadRgbdGpuInternal(dst, &depth_image, &rgb_image,
        depth_intrinsics, rgb_intrinsics, R, T, rgbToWorldR, rgbToWorldT, useCudaGLInterop);
}

void PointCloudLoader::loadRgbdGpuInternal(GaussianCloud& dst,
    const void* depth_mat_ptr, const void* rgb_mat_ptr,
    const glm::mat3& depth_intrinsics,
    const glm::mat3& rgb_intrinsics,
    const glm::mat3& R,
    const glm::vec3& T,
    const glm::mat3& rgbToWorldR,
    const glm::vec3& rgbToWorldT,
    bool useCudaGLInterop)
{
    const cv::Mat& depth_image = *static_cast<const cv::Mat*>(depth_mat_ptr);
    const cv::Mat& rgb_image = *static_cast<const cv::Mat*>(rgb_mat_ptr);

    if (depth_image.empty() || rgb_image.empty()) {
        std::cerr << "Error: Empty images passed to loadRgbdGpuInternal!" << std::endl;
        return;
    }

    int h_d = depth_image.rows;
    int w_d = depth_image.cols;
    int h_rgb = rgb_image.rows;
    int w_rgb = rgb_image.cols;
    int rgb_channels = rgb_image.channels();
    int total_pixels = h_d * w_d;

    // --- Persistent scratch GPU buffers (survive across calls) ---
    static uint16_t* s_d_depth = nullptr;
    static unsigned char* s_d_rgb = nullptr;
    static RgbdLoadOutputs s_outputs = {};
    static float* s_d_covX = nullptr;
    static float* s_d_covY = nullptr;
    static float* s_d_covZ = nullptr;
    static int s_alloc_pixels = 0;
    static int s_alloc_rgb = 0;
    static int s_alloc_cov = 0;

    // Re-allocate scratch input/output only when image dimensions change
    if (total_pixels != s_alloc_pixels) {
        if (s_d_depth)              cudaFree(s_d_depth);
        if (s_outputs.positions)    cudaFree(s_outputs.positions);
        if (s_outputs.normals)      cudaFree(s_outputs.normals);
        if (s_outputs.colors)       cudaFree(s_outputs.colors);
        if (s_outputs.opacities)    cudaFree(s_outputs.opacities);
        if (s_outputs.tangents)     cudaFree(s_outputs.tangents);
        if (s_outputs.depth_cam_points) cudaFree(s_outputs.depth_cam_points);
        if (s_outputs.scale_neus)   cudaFree(s_outputs.scale_neus);
        if (s_outputs.valid_count)  cudaFree(s_outputs.valid_count);

        cudaMalloc((void**)&s_d_depth, total_pixels * sizeof(uint16_t));
        cudaMalloc((void**)&s_outputs.positions, total_pixels * 4 * sizeof(float));
        cudaMalloc((void**)&s_outputs.normals, total_pixels * 4 * sizeof(float));
        cudaMalloc((void**)&s_outputs.colors, total_pixels * 3 * sizeof(float));
        cudaMalloc((void**)&s_outputs.opacities, total_pixels * sizeof(float));
        cudaMalloc((void**)&s_outputs.tangents, total_pixels * 3 * sizeof(float));
        cudaMalloc((void**)&s_outputs.depth_cam_points, total_pixels * 3 * sizeof(float));
        cudaMalloc((void**)&s_outputs.scale_neus, total_pixels * sizeof(float));
        cudaMalloc((void**)&s_outputs.valid_count, sizeof(int));
        s_alloc_pixels = total_pixels;
    }
    int rgb_bytes = h_rgb * w_rgb * rgb_channels;
    if (rgb_bytes != s_alloc_rgb) {
        if (s_d_rgb) cudaFree(s_d_rgb);
        cudaMalloc((void**)&s_d_rgb, rgb_bytes);
        s_alloc_rgb = rgb_bytes;
    }

    // Upload images to GPU
    cudaMemcpy(s_d_depth, depth_image.data, total_pixels * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_rgb, rgb_image.data, rgb_bytes, cudaMemcpyHostToDevice);

    // Fill kernel params
    RgbdLoadParams params = {};
    params.depth_image = s_d_depth;
    params.depth_width = w_d;
    params.depth_height = h_d;
    params.rgb_image = s_d_rgb;
    params.rgb_width = w_rgb;
    params.rgb_height = h_rgb;
    params.rgb_channels = rgb_channels;

    params.fx_depth = depth_intrinsics[0][0];
    params.fy_depth = depth_intrinsics[1][1];
    params.cx_depth = depth_intrinsics[2][0];
    params.cy_depth = depth_intrinsics[2][1];
    params.fx_rgb = rgb_intrinsics[0][0];
    params.fy_rgb = rgb_intrinsics[1][1];
    params.cx_rgb = rgb_intrinsics[2][0];
    params.cy_rgb = rgb_intrinsics[2][1];

    mat3ToRowMajor(R, params.R);
    params.T[0] = T.x;  params.T[1] = T.y;  params.T[2] = T.z;
    mat3ToRowMajor(rgbToWorldR, params.R_world);
    params.T_world[0] = rgbToWorldT.x;
    params.T_world[1] = rgbToWorldT.y;
    params.T_world[2] = rgbToWorldT.z;

    params.max_depth_mm = 2000.0f;
    params.border_left = 100;
    params.border_right = 100;
    params.border_top = 0;
    params.border_bottom = 0;

    // Launch unprojection kernel
    int num_valid = launchRgbdUnprojectKernel(params, s_outputs);

    if (num_valid == 0) {
        std::cerr << "Error: No valid points generated from RGBD images!" << std::endl;
        return;
    }

    // --- Decide whether we can reuse existing dst allocations ---
    const bool sameSize = dst.initialized && dst.num_gaussians == num_valid;

    if (!sameSize) {
        dst.freeRawCudaBuffers();
        dst.clearCpuData();
        dst.initialized = false;
    }
    dst.num_gaussians = num_valid;

    // Re-allocate covariance scratch when point count changes
    if (num_valid != s_alloc_cov) {
        if (s_d_covX) cudaFree(s_d_covX);
        if (s_d_covY) cudaFree(s_d_covY);
        if (s_d_covZ) cudaFree(s_d_covZ);
        cudaMalloc((void**)&s_d_covX, num_valid * 4 * sizeof(float));
        cudaMalloc((void**)&s_d_covY, num_valid * 4 * sizeof(float));
        cudaMalloc((void**)&s_d_covZ, num_valid * 4 * sizeof(float));
        s_alloc_cov = num_valid;
    }

    // Launch covariance kernel
    float R_row[9], Rw_row[9];
    mat3ToRowMajor(R, R_row);
    mat3ToRowMajor(rgbToWorldR, Rw_row);
    launchDepthCovarianceKernel(
        s_outputs.depth_cam_points, s_outputs.tangents,
        params.fx_depth, params.fy_depth,
        R_row, params.T, Rw_row,
        s_d_covX, s_d_covY, s_d_covZ,
        num_valid);

    // ---- Upload data via storeOrUpdateData (reuses GL buffers when size matches) ----

    // Positions
    dst.positions_cpu.resize(num_valid);
    cudaMemcpy(dst.positions_cpu.data(), s_outputs.positions, num_valid * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    dst.positions.storeOrUpdateData(dst.positions_cpu.data(), num_valid, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Normals
    dst.normals_cpu.resize(num_valid);
    cudaMemcpy(dst.normals_cpu.data(), s_outputs.normals, num_valid * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    dst.normals.storeOrUpdateData(dst.normals_cpu.data(), num_valid, 4 * sizeof(float), 0, useCudaGLInterop, false, true);

    // Opacities
    dst.opacities_cpu.assign(num_valid, 1.0f);
    dst.opacities.storeOrUpdateData(dst.opacities_cpu.data(), num_valid, sizeof(float), 0, useCudaGLInterop, false, true);

    // SH coefficients
    const float SH_C0 = 0.28209479177387814f;
    std::vector<float> colors_cpu(num_valid * 3);
    cudaMemcpy(colors_cpu.data(), s_outputs.colors, num_valid * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++) {
        std::vector<float> sh(num_valid * 16, 0.0f);
        for (int k = 0; k < num_valid; k++)
            sh[k * 16] = (colors_cpu[k * 3 + i] - 0.5f) / SH_C0;
        dst.sh_coeffs[i].storeOrUpdateData(sh.data(), num_valid, 16 * sizeof(float), 0, useCudaGLInterop, false, true);
    }

    // Covariance
    dst.covX_cpu.resize(num_valid);
    dst.covY_cpu.resize(num_valid);
    dst.covZ_cpu.resize(num_valid);
    cudaMemcpy(dst.covX_cpu.data(), s_d_covX, num_valid * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.covY_cpu.data(), s_d_covY, num_valid * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.covZ_cpu.data(), s_d_covZ, num_valid * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int row = 0; row < 3; row++) {
        const glm::vec4* src_data = (row == 0 ? dst.covX_cpu.data() : (row == 1 ? dst.covY_cpu.data() : dst.covZ_cpu.data()));
        dst.covariance[row].storeOrUpdateData(src_data, num_valid, 4 * sizeof(float), 0, useCudaGLInterop, false, true);
    }

    // SDF
    dst.sdf_cpu.assign(num_valid, 0.0f);
    dst.sdf.storeOrUpdateData(dst.sdf_cpu.data(), num_valid, sizeof(float), 0, useCudaGLInterop, false, true);

    // NeuS sharpness (computed per-point in the unproject kernel)
    dst.scale_neus_cpu.resize(num_valid);
    cudaMemcpy(dst.scale_neus_cpu.data(), s_outputs.scale_neus, num_valid * sizeof(float), cudaMemcpyDeviceToHost);
    dst.scale_neus.storeOrUpdateData(dst.scale_neus_cpu.data(), num_valid, sizeof(float), 0, useCudaGLInterop, false, true);

    // ---- Derived / scratch buffers: only allocate when size changes ----
    if (!sameSize) {
        dst.visible_gaussians_counter.storeData(nullptr, 1, sizeof(int), 0, useCudaGLInterop, false, true);
        dst.gaussians_depths.storeData(nullptr, num_valid, sizeof(float), 0, useCudaGLInterop, true, true);
        dst.gaussians_indices.storeData(nullptr, num_valid, sizeof(int), 0, useCudaGLInterop, true, true);
        dst.sorted_depths.storeData(nullptr, num_valid, sizeof(float), 0, useCudaGLInterop, true, true);
        dst.sorted_gaussian_indices.storeData(nullptr, num_valid, sizeof(int), 0, useCudaGLInterop, true, true);

        dst.bounding_boxes.storeData(nullptr, num_valid, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.conic_opacity.storeData(nullptr, num_valid, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.eigen_vecs.storeData(nullptr, num_valid, 2 * sizeof(float), 0, useCudaGLInterop, true, true);
        dst.predicted_colors.storeData(nullptr, num_valid, 4 * sizeof(float), 0, useCudaGLInterop, true, true);
    }

    dst.initialized = true;
}