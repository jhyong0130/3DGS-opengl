//
//  CVT.hpp
//  
//
//  Created by Diego Thomas on 2025/07/18.
//

#ifndef __CVT_H
#define __CVT_H
#pragma once
#include "Utilities.h"
#include <nanoflann.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <set>
#include <map>
#include <algorithm>
#include "CVT_update_cpp.hpp"
#include <future>
#include "GUI2D.hpp"

/*#include "RenderingBase/GLBuffer.h"
#include "RenderingBase/Shader.h"
#include "RenderingBase/GLShaderLoader.h"
#include "RenderingBase/Camera.h"
#include "RenderingBase/VAO.h"*/
#include "RenderingBase/GLTimer.h"

#include "KNN_cuda.cuh"
#include "CVTupdate.cuh"
#include "Sort.cuh"
#include <cuda_gl_interop.h>  // Required for cudaGraphicsGLRegisterBuffer

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef K::Point_3 Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Vertex_iterator Vertex_iterator;
typedef Delaunay::Finite_vertices_iterator Finite_vertex_iterator;


struct Tetrahedron {
    glm::vec3 v0, v1, v2, v3;
};


// Define the point cloud structure
struct PointCloud {
    std::vector<float3> pts;
    std::vector<glm::vec3> fork_pts;
    std::vector<float> float_pts;
    std::vector<GLuint> indices_pt;
    std::vector<GLuint> indices;
    std::vector<float> rgba;
    std::vector<unsigned char> flags;

    // Gaussian splatting attributes
    std::vector<glm::vec3> f_dc;      // f_dc_0, f_dc_1, f_dc_2
    std::vector<std::array<float, 45>> f_rest; // f_rest_0 ... f_rest_44
    std::vector<float> sdf;   // opacity
    std::vector<glm::vec3> scale;     // scale_0, scale_1, scale_2
    std::vector<glm::vec4> rot;       // rot_0 ... rot_3
    std::vector<float> cov;
    
    std::vector<Tetrahedron> isocahedron;
    std::vector<glm::vec3> isocahedron_rays;

    std::vector<unsigned int> ret_index;
    std::vector<float> out_dist_sqr;

    std::vector<unsigned int> fork_ret_index;

    GLuint shader_programme, merge_programme, shader_float_programme;
    GLuint vbo_in, vbo_out, vbo_attrib_in, vbo_float_out, vbo_attrib_out, 
        vao, tbo, tbo_out, tbo_attrib_out, tex, tex_out, tex_attributes, vbo_indx, vbo_indx_pt;
    GLuint tbo_adjacents, tex_adjacents;

    GLuint shader_3DGS_programme;
    GLuint shader_programme_normalize;
    //vertex buffer objects for handling Gaussian Splatting
    GLuint vao_GS, vbo_sdf, vbo_dc, vbo_cov, ebo;
    GLuint accumFBO, accumTex;
    GLuint quadVAO, quadVBO;
    float quadVertices[30] = {
        // positions             // texture coords
        1.0f, -1.0f, 0.0f,        1.0, 0.0,
        1.0f, 1.0f, 0.0f,        1.0, 1.0,
        -1.0, -1.0, 0.0f,       0.0, 0.0,
        -1.0, 1.0, 0.0f,       0.0, 1.0,
        -1.0, -1.0, 0.0f,       0.0, 0.0,
        1.0, 1.0, 0.0f,         1.0, 1.0
    };

    GLfloat thresh_vals[2] {-0.2, 0.2};

    cudaGraphicsResource* cuda_vbo_in_resource;
    cudaGraphicsResource* cuda_vbo_cov_resource;
    cudaGraphicsResource* cuda_vbo_sdf_resource;
    uint4* d_adjacencies;
    uint4* d_adjacencies_delaunay;
    float* d_cov;
    float* d_sdf;
    unsigned char* d_flags;
    float* d_thresh_vals;

    lbvh_tree* knn_tree;
    uint32_t* morton_codes;
    uint32_t* sorted_indices;
    uint32_t* indices_out;
    float* distances_out;
    uint32_t* n_neighbors_out;
    
    Delaunay dt;
    std::future<void> delaunay_future;

    int KVal = 32;
    int KVal_d = 0;
    int KVal_d_prev = 0;
    int fork_KVal_d = 0;
    int NbPts = 0;
    int CurrNbPts = 0;
    int buff_symmetry = 0;

    int delaunay_done = 0;

    enum OPERATIONS {
        PREDICT_COLORS_ALL,
        DRAW_AS_POINTS,
        TEST_VISIBILITY,
        SORT,
        COMPUTE_BOUNDING_BOXES,
        PREDICT_COLORS_VISIBLE,
        DRAW_AS_QUADS,
        NUM_OPS
    };

    QueryBuffer timers[OPERATIONS::NUM_OPS];

    PointCloud() {
        knn_tree = new lbvh_tree(32, false, false, 10.0f, true, KVal);
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return CurrNbPts; //pts.size();
    }

    inline size_t get_point_count() const {
        return CurrNbPts; //pts.size();
    }

    // Returns the dim'th component of the idx'th point in the class
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return (dim == 0) ? pts[idx].x : (dim == 1) ? pts[idx].y: pts[idx].z;
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

    /*inline float* points() {
        return pts.data();
    }*/

    inline float* colors() {
        return rgba.data();
    }

    // Randomly fill point cloud with points in [0, 1)
    void generateRandom(size_t N) {
        pts.clear(); indices_pt.clear(); indices.clear(); rgba.clear(); flags.clear();
        ret_index.clear(); out_dist_sqr.clear();
        
        NbPts = N;
        CurrNbPts = N;
        
        if (N <= 1000) {
            buff_symmetry = 1000;
        }
        else if (N <= 10000) {
            buff_symmetry = 5000;
        }
        else if (N <= 100000) {
            buff_symmetry = 10000;
        } 
        else {
            buff_symmetry = 50000;
        }

        buff_symmetry = 0;
        
        
        pts.resize(N+buff_symmetry);
        fork_pts.resize(N+buff_symmetry);
        float_pts.resize(N+buff_symmetry);
        indices_pt.resize(N + buff_symmetry);
        indices.resize(20*(N+buff_symmetry));
        rgba.resize(4*(N+buff_symmetry));
        flags.resize(N+buff_symmetry);

        // Gaussians attributes
        f_dc.resize(N + buff_symmetry);      // f_dc_0, f_dc_1, f_dc_2
        f_rest.resize(N + buff_symmetry); // f_rest_0 ... f_rest_44
        sdf.resize(N + buff_symmetry);   // opacity
        scale.resize(N + buff_symmetry);     // scale_0, scale_1, scale_2
        rot.resize(N + buff_symmetry);       // rot_0 ... rot_3
        cov.resize(9*(N + buff_symmetry));       // rot_0 ... rot_3

        for (size_t i = 0; i < N; ++i) {
            pts[i].x = 1.0f - 2.0f*static_cast<float>(rand()) / RAND_MAX;
            pts[i].y = 1.0f - 2.0f*static_cast<float>(rand()) / RAND_MAX;
            pts[i].z = 1.0f - 2.0f*static_cast<float>(rand()) / RAND_MAX;

            indices_pt[i] = i;
            for (int j = 0; j < 20; j++) {
                indices[20*i + j] = 20 * i + j;
            }

            rgba[4*i] = 1.0f;
            rgba[4*i+1] = 1.0f;
            rgba[4*i+2] = 1.0f;
            rgba[4*i+3] = 1.0f;

            sdf[i] = h_SDF_Torus(pts[i], 0.6f, 0.4f);
            cov[9 * i] = 0.001f;
            cov[9 * i + 4] = 0.001f;
            cov[9 * i + 8] = 0.001f;

            f_dc[i] = glm::vec3(1.0f, 1.0f, 1.0f);
        }

        ret_index.resize((KVal+KVal_d)*(N+buff_symmetry));
        fork_ret_index.resize((KVal+KVal_d)*(N+buff_symmetry));
        out_dist_sqr.resize((KVal+KVal_d)*(N+buff_symmetry));

    }

    void buildIsocahedron() {
        // Golden ratio
        float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
        float a = 1.0f / std::sqrt(1.0f + phi * phi);
        float b = phi * a;

        // 12 vertices of an icosahedron (scaled to unit sphere)
        std::vector<glm::vec3> vertices = {
            { 0,  a,  b}, { 0, -a,  b}, { 0,  a, -b}, { 0, -a, -b},
            { a,  b, 0}, {-a,  b, 0}, { a, -b, 0}, {-a, -b, 0},
            { b, 0,  a}, {-b, 0,  a}, { b, 0, -a}, {-b, 0, -a}
        };

        // Normalize all vertices to ensure they're on the unit sphere
        for (auto& v : vertices) v = glm::length(v) > 0.0 ? glm::normalize(v) : v;

        // Faces of the icosahedron (as triangles)
        std::vector<std::array<int, 3>> faces = {
            {0, 8, 4}, {0, 5, 4}, {0, 8, 1}, {0, 9, 1}, {0, 9, 5},
            {1, 9, 7}, {1, 7, 6}, {1, 6, 8}, {8, 6, 10}, {8, 4, 10},
            {4, 10, 2}, {4, 2, 5}, {2, 10, 3}, {2, 3, 11}, {2, 11, 5},
            {5, 11, 9}, {11, 9, 7}, {11, 7, 3}, {3, 6, 10}, {3, 6, 7}
        };
        
        glm::vec3 center(0.0f);
        int centerIndex = vertices.size();
        vertices.push_back(center); // index = 12

        std::vector<std::array<int, 4>> tetra;
        for (const auto& face : faces) {
            Tetrahedron t = {
                vertices[face[0]],
                vertices[face[1]],
                vertices[face[2]],
                center
            };
            isocahedron.push_back(t);
            std::array<int, 4> arr = {face[0], face[1], face[2], centerIndex};
            tetra.push_back(arr);
        }

        for (const auto& t : isocahedron) {
            isocahedron_rays.push_back(glm::normalize(t.v0-t.v3));
            isocahedron_rays.push_back(glm::normalize(t.v1-t.v3));
            isocahedron_rays.push_back(glm::normalize(t.v2-t.v3));
            std::cout << glm::normalize(t.v0-t.v3).x << ", " << glm::normalize(t.v0-t.v3).y << ", " << glm::normalize(t.v0-t.v3).z << std::endl;
            std::cout << glm::normalize(t.v1-t.v3).x << ", " << glm::normalize(t.v1-t.v3).y << ", " << glm::normalize(t.v1-t.v3).z << std::endl;
            std::cout << glm::normalize(t.v2-t.v3).x << ", " << glm::normalize(t.v2-t.v3).y << ", " << glm::normalize(t.v2-t.v3).z << std::endl;
        }
        
        // save as ply
        // Write to ASCII PLY
        std::ofstream out("isocahedron.ply");
        if (!out) {
            std::cerr << "Failed to open output file.\n";
            return;
        }

        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "element vertex " << vertices.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "element face " << 4*tetra.size() << "\n";
        out << "property list uchar int vertex_indices\n";
        out << "end_header\n";

        for (const auto& v : vertices) {
            out << v.x << " " << v.y << " " << v.z << "\n";
        }

        for (const auto& tet : tetra) {
            out << "3 " << tet[0] << " " << tet[1] << " " << tet[2] << "\n";
            out << "3 " << tet[0] << " " << tet[1] << " " << tet[3] << "\n";
            out << "3 " << tet[0] << " " << tet[2] << " " << tet[3] << "\n";
            out << "3 " << tet[1] << " " << tet[2] << " " << tet[3] << "\n";
        }

        out.close();
        std::cout << "PLY file written successfully.\n";
        return;

    }

    void buildShaders() {
        GLuint v_shader = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("CVTupdate.glsl")));

        // link shaders
        shader_programme = glCreateProgram();
        glAttachShader(shader_programme, v_shader);

        // Declare output varyings before linking
        const char* varyings[] = { "outPosition", "outAttributes" };
        glTransformFeedbackVaryings(shader_programme, 2, varyings, GL_SEPARATE_ATTRIBS);


        glLinkProgram(shader_programme);
        // check for linking errors
        int success;
        char infoLog[512];
        glGetProgramiv(shader_programme, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader_programme, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::shader_programme::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(v_shader);


        GLuint m_float_shader = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("MergeCentroids.glsl")));

        // link shaders
        merge_programme = glCreateProgram();
        glAttachShader(merge_programme, m_float_shader);

        // Declare output varyings before linking
        const char* varyings_merge[] = { "outPosition", "outAttributes" };
        glTransformFeedbackVaryings(merge_programme, 2, varyings_merge, GL_SEPARATE_ATTRIBS);


        glLinkProgram(merge_programme);
        // check for linking errors
        glGetProgramiv(merge_programme, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(merge_programme, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::shader_programme::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(m_float_shader);


        GLuint v_float_shader = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("Flatten.glsl")));

        // link shaders
        shader_float_programme = glCreateProgram();
        glAttachShader(shader_float_programme, v_float_shader);

        // Declare output varyings before linking
        const char* varyings_float[] = { "outPosition" };
        glTransformFeedbackVaryings(shader_float_programme, 1, varyings_float, GL_SEPARATE_ATTRIBS);


        glLinkProgram(shader_float_programme);
        // check for linking errors
        glGetProgramiv(shader_float_programme, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader_float_programme, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::shader_programme::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(v_float_shader);
    }

    void buildShaders3DGS() {
        GLuint v_shader = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("3DGS_vertex.glsl")));
        GLuint g_shader = compileShader(GL_GEOMETRY_SHADER, loadShaderSource(string(SHADERS_PATH) + string("3DGS_geo.glsl")));
        GLuint f_shader = compileShader(GL_FRAGMENT_SHADER, loadShaderSource(string(SHADERS_PATH) + string("3DGS_frag.glsl")));
        GLuint v_shader_normalise = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("SimpleVertexShader.glsl")));
        GLuint f_shader_normalise = compileShader(GL_FRAGMENT_SHADER, loadShaderSource(string(SHADERS_PATH) + string("Normalise.glsl")));

        // link shaders
        shader_3DGS_programme = glCreateProgram();
        glAttachShader(shader_3DGS_programme, v_shader);
        glAttachShader(shader_3DGS_programme, g_shader);
        glAttachShader(shader_3DGS_programme, f_shader);
        glLinkProgram(shader_3DGS_programme);
        // check for linking errors
        int success;
        char infoLog[512];
        glGetProgramiv(shader_3DGS_programme, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader_3DGS_programme, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::shader_3DGS_programme::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(v_shader);
        glDeleteShader(g_shader);
        glDeleteShader(f_shader);

        shader_programme_normalize = glCreateProgram();
        glAttachShader(shader_programme_normalize, v_shader_normalise);
        glAttachShader(shader_programme_normalize, f_shader_normalise);
        glLinkProgram(shader_programme_normalize);
        // check for linking errors
        glGetProgramiv(shader_programme_normalize, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader_programme_normalize, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::shader_programme_normalize::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(v_shader_normalise);
        glDeleteShader(f_shader_normalise);

        return;
    }

    void buildVBO() {
        std::cout << "pts.size() " << pts.size() << std::endl;
        #ifdef _CUDA_ENABLED_
            if ((KVal + KVal_d) % 4 > 0) {
                std::cout << "ERRROR number of adjacencies must be multiple of 4" << std::endl;
                exit(-1);
            }
            cudaMalloc(&d_adjacencies, (KVal + KVal_d) * pts.size() * sizeof(uint)); // KVal + KVal_d must be multiple of 4
            cudaMalloc(&d_adjacencies_delaunay, KVal_d * pts.size() * sizeof(uint)); // KVal + KVal_d must be multiple of 4
            //cudaMalloc(&d_cov, 9*pts.size() * sizeof(float)); // KVal + KVal_d must be multiple of 4
            cudaMalloc(&d_flags, pts.size() * sizeof(unsigned char)); // KVal + KVal_d must be multiple of 4
            cudaMalloc(&d_thresh_vals, 2 * sizeof(float)); // KVal + KVal_d must be multiple of 4

            cudaMalloc(&morton_codes, pts.size() * sizeof(uint64_t));
            cudaMalloc(&sorted_indices, pts.size() * sizeof(uint32_t));
            cudaMalloc(&indices_out, KVal * pts.size() * sizeof(uint32_t));
            cudaMalloc(&distances_out, KVal * pts.size() * sizeof(float));
            cudaMalloc(&n_neighbors_out, pts.size() * sizeof(uint32_t));
        #endif

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Input VBO
        glGenBuffers(1, &vbo_in);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float3), pts.data(), GL_DYNAMIC_READ);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // Input VBO
        glGenBuffers(1, &vbo_indx);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx);
        glBufferData(GL_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, (void*)0);

        // Input VBO
        glGenBuffers(1, &vbo_indx_pt);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx_pt);
        glBufferData(GL_ARRAY_BUFFER, indices_pt.size() * sizeof(GLuint), indices_pt.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribIPointer(2, 1, GL_UNSIGNED_INT, 0, (void*)0);
        
        #ifdef _CUDA_ENABLED_
            cudaGraphicsGLRegisterBuffer(&cuda_vbo_in_resource, vbo_in, cudaGraphicsMapFlagsWriteDiscard);
        #endif

        glGenBuffers(1, &vbo_attrib_in);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_attrib_in);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float), nullptr, GL_DYNAMIC_READ);

        // Output VBO (same size, initially empty)
        glGenBuffers(1, &vbo_out);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_out);
        glBufferData(GL_ARRAY_BUFFER, 20 * pts.size() * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);
        
        glGenBuffers(1, &vbo_attrib_out);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_attrib_out);
        glBufferData(GL_ARRAY_BUFFER, 20 * pts.size() * sizeof(float), nullptr, GL_STATIC_DRAW);

        glGenBuffers(1, &vbo_float_out);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_float_out);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * 3 * sizeof(float), nullptr, GL_STATIC_READ);

        glBindVertexArray(0);

        // Generate TBO
        glGenBuffers(1, &tbo);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo);
        glBufferData(GL_TEXTURE_BUFFER, pts.size() * sizeof(float3), pts.data(), GL_STATIC_DRAW);

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_BUFFER, tex);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo); // float format

        // Generate TBO
        glGenBuffers(1, &tbo_out);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_out);
        glBufferData(GL_TEXTURE_BUFFER, 20 * pts.size() * sizeof(glm::vec4), nullptr, GL_STREAM_COPY);

        glGenTextures(1, &tex_out);
        glBindTexture(GL_TEXTURE_BUFFER, tex_out);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbo_out); // float format

        // Generate TBO
        glGenBuffers(1, &tbo_attrib_out);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_attrib_out);
        glBufferData(GL_TEXTURE_BUFFER, 20 * pts.size() * sizeof(float), nullptr, GL_STREAM_COPY);

        glGenTextures(1, &tex_attributes);
        glBindTexture(GL_TEXTURE_BUFFER, tex_attributes);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo_attrib_out); // float format



        glGenBuffers(1, &tbo_adjacents);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_adjacents);
        glBufferData(GL_TEXTURE_BUFFER, (KVal+KVal_d) * pts.size() * sizeof(unsigned int), ret_index.data(), GL_STATIC_DRAW);

        glGenTextures(1, &tex_adjacents);
        glBindTexture(GL_TEXTURE_BUFFER, tex_adjacents);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, tbo_adjacents); // float format



        glGenVertexArrays(1, &vao_GS);
        glBindVertexArray(vao_GS);

        // Input VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &vbo_sdf);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sdf);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float), sdf.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &vbo_dc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_dc);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3), f_dc.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &vbo_cov);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cov);
        glBufferData(GL_ARRAY_BUFFER, 9*pts.size() * sizeof(float), cov.data(), GL_STATIC_DRAW);
        // Row 0
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(0));
        // Row 1
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
        // Row 2
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));

        glGenBuffers(1, &ebo);
        // Index buffer (Element Buffer Object)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, pts.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);


        #ifdef _CUDA_ENABLED_
                cudaGraphicsGLRegisterBuffer(&cuda_vbo_cov_resource, vbo_cov, cudaGraphicsMapFlagsWriteDiscard);
                cudaGraphicsGLRegisterBuffer(&cuda_vbo_sdf_resource, vbo_sdf, cudaGraphicsMapFlagsWriteDiscard);
        #endif

        glBindVertexArray(0);

    }

    void buildFBO(int width_tex, int height_tex) {
        // Create a floating-point texture to accumulate color + alpha
        glGenTextures(1, &accumTex);
        glBindTexture(GL_TEXTURE_2D, accumTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width_tex, height_tex, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Attach to FBO
        glGenFramebuffers(1, &accumFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);

        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            5 * sizeof(float),                  // stride
            (void*)0            // array buffer offset
        );

        // 3rd attribute buffer : vertices
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            2,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            5 * sizeof(float),                  // stride
            (void*)12            // array buffer offset
        );

        glBindVertexArray(0);
    }

    void updateCVT() {
        /*vector<glm::vec3> tmp = CVTUpdate_cpp(pts, ret_index, KVal);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        //glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
        void* ptr_out = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr_out, tmp.data(), pts.size() * sizeof(glm::vec3));
        glUnmapBuffer(GL_ARRAY_BUFFER);
        
        
        glBindBuffer(GL_TEXTURE_BUFFER, tbo);
        ptr_out = glMapBuffer(GL_TEXTURE_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr_out, tmp.data(), pts.size() * sizeof(glm::vec3));
        glUnmapBuffer(GL_TEXTURE_BUFFER);
        
        memcpy(pts.data(), tmp.data(), pts.size() * sizeof(glm::vec3));
        
        return;*/

        glUseProgram(shader_programme);
        glBindVertexArray(vao);
        
        GLint posLoc = glGetAttribLocation(shader_programme, "inIndex");
        glEnableVertexAttribArray(posLoc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx);
        glVertexAttribIPointer(posLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);
        //glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        //glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);


        GLint KValLoc = glGetUniformLocation(shader_programme, "K_val");
        glUniform1i(KValLoc, (KVal+KVal_d));


        GLint ThreshLoc = glGetUniformLocation(shader_programme, "thresh_vals");
        glUniform1fv(ThreshLoc, 2, thresh_vals);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, tex);
        glUniform1i(glGetUniformLocation(shader_programme, "vertices"), 0);


        glBindBuffer(GL_TEXTURE_BUFFER, tbo_adjacents);
        void* ptr = glMapBuffer(GL_TEXTURE_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, ret_index.data(), (KVal+KVal_d) * CurrNbPts * sizeof(unsigned int));
        glUnmapBuffer(GL_TEXTURE_BUFFER);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, tex_adjacents);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, tbo_adjacents); // float format
        glUniform1i(glGetUniformLocation(shader_programme, "adjacents"), 1);

        // Bind output buffer
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo_out);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, vbo_attrib_out);

        // Disable rasterization
        glEnable(GL_RASTERIZER_DISCARD);

        // Perform transform feedback
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 20*CurrNbPts);
        glEndTransformFeedback();

        glDisable(GL_RASTERIZER_DISCARD);

        // Copy 256 bytes from offset 0 in source to offset 0 in destination
        glBindBuffer(GL_COPY_READ_BUFFER, vbo_out);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tbo_out);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
            0, 0, 20*CurrNbPts * sizeof(glm::vec4));

        glBindBuffer(GL_COPY_READ_BUFFER, vbo_attrib_out);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tbo_attrib_out);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
            0, 0, 20*CurrNbPts * sizeof(float));

        MergeCentroids();

        //CurrNbPts = NbPts;

        // Copy 256 bytes from offset 0 in source to offset 0 in destination
        glBindBuffer(GL_COPY_READ_BUFFER, vbo_in);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 
                            0, 0, CurrNbPts * sizeof(glm::vec3));

        auto start = std::chrono::high_resolution_clock::now();
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        void* srcPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        memcpy(pts.data(), srcPtr, CurrNbPts * sizeof(glm::vec3));
        glUnmapBuffer(GL_ARRAY_BUFFER);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //std::cout << "A " << duration << std::endl;


        start = std::chrono::high_resolution_clock::now();
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_attrib_in);
        srcPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        memcpy(flags.data(), srcPtr, CurrNbPts * sizeof(float));
        glUnmapBuffer(GL_ARRAY_BUFFER);

        // Record end time
        end = std::chrono::high_resolution_clock::now();

        // Compute duration in milliseconds
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //std::cout << duration << std::endl;

        
        
        /*for (const auto& f : flags) {
            std::cout << "flag " << f << "\n";
        }*/
        
        /*glm::vec3* dstPtr = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // Read back results
        glm::vec3* newData = (glm::vec3*)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
        for (size_t i = 0; i < 11; ++i) {
            //std::cout << "TMP Pos " << i << ": " << tmp[i].x << ", " << tmp[i].y << ", " << tmp[i].z << "\n";
            std::cout << "New Pos " << i << ": " << newData[i].x << ", " << newData[i].y << ", " << newData[i].z << "\n";
            std::cout << "Adj " << i << ": " << ret_index[16*i] << ", " << ret_index[16*i + 1] << ", " << ret_index[16*i + 2] << "\n";
        }
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
        glUnmapBuffer(GL_ARRAY_BUFFER);*/
    }

    void updateCVT_CUDA() {
        auto start = std::chrono::high_resolution_clock::now();

        /*if (KVal_d_prev < KVal_d) {
            if ((KVal + KVal_d) % 4 > 0) {
                std::cout << "ERRROR number of adjacencies must be multiple of 4" << std::endl;
                exit(-1);
            }
            cudaMalloc(&d_adjacencies, (KVal + KVal_d) * pts.size() * sizeof(uint)); // KVal + KVal_d must be multiple of 4
            KVal_d_prev = KVal_d;
        }*/
        //cudaMemcpy(d_adjacencies, ret_index.data(), (KVal + KVal_d) * pts.size() * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_thresh_vals, thresh_vals, 2 * sizeof(float), cudaMemcpyHostToDevice);

        float3* dptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_vbo_in_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_in_resource);

        cudaGraphicsMapResources(1, &cuda_vbo_cov_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_cov, &num_bytes, cuda_vbo_cov_resource);

        cudaGraphicsMapResources(1, &cuda_vbo_sdf_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_sdf, &num_bytes, cuda_vbo_sdf_resource);

        cudaMemset(d_flags, 0, pts.size() * sizeof(unsigned char));

        updateCVT_cu(d_sdf, dptr, d_adjacencies, d_cov, d_flags, KVal + KVal_d, d_thresh_vals, CurrNbPts);

        cudaMemcpy(pts.data(), dptr, pts.size() * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(flags.data(), d_flags, pts.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        /*cudaMemcpy(cov.data(), d_cov, 9 * pts.size() * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 500; i < 510; i++) {
            std::cout << i << std::endl;
            std::cout << cov[9*i] << ", " << cov[9 * i + 1] << ", " << cov[9 * i + 2] << std::endl;
            std::cout << cov[9 * i + 3] << ", " << cov[9 * i + 4] << ", " << cov[9 * i + 5] << std::endl;
            std::cout << cov[9 * i + 6] << ", " << cov[9 * i + 7] << ", " << cov[9 * i + 8] << std::endl;

        }*/

        cudaGraphicsUnmapResources(1, &cuda_vbo_sdf_resource, 0);
        cudaGraphicsUnmapResources(1, &cuda_vbo_cov_resource, 0);
        cudaGraphicsUnmapResources(1, &cuda_vbo_in_resource, 0);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "updateCVT_CUDA -> " <<  duration.count() << " ---- with " << CurrNbPts <<  " points" << std::endl;

        //CurrNbPts = NbPts;
    }

    void MergeCentroids() {
        glUseProgram(merge_programme);
        glBindVertexArray(vao);

        GLint posLoc = glGetAttribLocation(merge_programme, "inIndex");
        glEnableVertexAttribArray(posLoc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx_pt);
        glVertexAttribIPointer(posLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);

        GLint ThreshLoc = glGetUniformLocation(merge_programme, "thresh_vals");
        glUniform1fv(ThreshLoc, 2, thresh_vals);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, tex);
        glUniform1i(glGetUniformLocation(merge_programme, "vertices_in"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, tex_out);
        glUniform1i(glGetUniformLocation(merge_programme, "vertices"), 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, tex_attributes);
        glUniform1i(glGetUniformLocation(merge_programme, "attributes"), 2);

        // Bind output buffer
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo_in);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, vbo_attrib_in);

        // Disable rasterization
        glEnable(GL_RASTERIZER_DISCARD);

        // Perform transform feedback
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, CurrNbPts);
        glEndTransformFeedback();

        glDisable(GL_RASTERIZER_DISCARD);
    }

    void flattenPoints() {
        glUseProgram(shader_float_programme);
        glBindVertexArray(vao);
        
        GLint posLoc = glGetAttribLocation(shader_float_programme, "in_position");
        glEnableVertexAttribArray(posLoc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_in);
        glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        // Bind output buffer
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo_float_out);

        // Disable rasterization
        glEnable(GL_RASTERIZER_DISCARD);

        // Perform transform feedback
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, CurrNbPts);
        glEndTransformFeedback();

        glDisable(GL_RASTERIZER_DISCARD);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_out);
        void* srcPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        memcpy(float_pts.data(), srcPtr, CurrNbPts * 3 * sizeof(float));
        glUnmapBuffer(GL_ARRAY_BUFFER);
       
    }
    
    void KNN_cuda() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        /*if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            std::cout << "Device " << device << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
            std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
            std::cout << std::endl;
        }*/


        int currentDevice = -1;
        // Get the currently active CUDA device
        err = cudaGetDevice(&currentDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(err) << std::endl;
            // Handle error (maybe set to a default device)
        }

        // If you want to explicitly set a device (say device 0)
        int deviceToUse = 0;
        if (currentDevice != deviceToUse) {
            err = cudaSetDevice(deviceToUse);
            if (err != cudaSuccess) {
                std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
                // Handle error
            }
        }

        glFinish();
        auto start_knn = std::chrono::high_resolution_clock::now();
        float3* dptr;
        size_t num_bytes;
        if (cudaGraphicsMapResources(1, &cuda_vbo_in_resource, 0) != cudaSuccess) {
            std::cerr << "Failed to map CUDA graphics resource" << std::endl;
            return;
        }
        if (cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_in_resource) != cudaSuccess) {
            std::cerr << "Failed to get mapped pointer from CUDA graphics resource" << std::endl;
            return;
        }
        assert(num_bytes >= CurrNbPts * sizeof(float3));


        knn_tree->Build(dptr, CurrNbPts);
        knn_tree->Query_KNN(dptr, morton_codes, sorted_indices, indices_out, distances_out, n_neighbors_out, CurrNbPts);

        cudaGraphicsUnmapResources(1, &cuda_vbo_in_resource, 0);

        knn_tree->Remap2uint4(d_adjacencies, d_adjacencies_delaunay, indices_out, CurrNbPts, KVal, KVal_d);

        //cudaMemcpy(d_adjacencies, indices_out, KVal * CurrNbPts * sizeof(uint), cudaMemcpyDeviceToDevice);

        /*std::cout << "CPU " << std::endl;
        for (int i = 0; i < 1; i++) {
            std::cout << " pts " << i << " -> " << pts[i].x << ", " << pts[i].y << ", " << pts[i].z << std::endl;
            for (int j = 0; j < KVal; j++) {
                std::cout << j << " -> " << ret_index[i * KVal + j] << ", " << out_dist_sqr[i * KVal + j] << std::endl;
            }
        }

        std::cout << length(pts[0] - pts[ret_index[3]]) << std::endl;

        cudaMemcpy(ret_index.data(), d_adjacencies, KVal * CurrNbPts * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dist_sqr.data(), distances_out, KVal * CurrNbPts * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "GPU " << std::endl;
        for (int i = 0; i < 1; i++) {
            std::cout << " pts " << i << " -> " << pts[i].x << ", " << pts[i].y << ", " << pts[i].z << std::endl;
            for (int j = 0; j < KVal; j++) {
                std::cout << j << " -> " << ret_index[i * KVal + j] << ", " << out_dist_sqr[i * KVal + j] << std::endl;
            }
        }

        std::cout << length(pts[0] - pts[ret_index[3]]) << std::endl;*/

        auto end_knn = std::chrono::high_resolution_clock::now();
        auto duration_knn = std::chrono::duration_cast<std::chrono::milliseconds>(end_knn - start_knn);
        std::cout << "KNN_CUDA -> " << duration_knn.count() << std::endl;
        //int tmp;
        //std::cin >> tmp;
    }

    void doDelaunay() {
        auto start = std::chrono::high_resolution_clock::now();
        
        dt.clear();
        
        int id_count = 0;
        for (const auto& point : fork_pts) {
            dt.insert(Point(point.x, point.y, point.z));
            id_count++;
            if (id_count == NbPts)
                break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Delaunay computed in " << duration.count() << endl;
        
        start = std::chrono::high_resolution_clock::now();
        
        std::map<Vertex_handle, std::size_t> vertex_indices;
        std::size_t index = 0;

        for (Finite_vertex_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
            vertex_indices[vit] = index++;
        }
        
        // Create adjacency map
        size_t max_nb_adj = 0;
        std::map<std::size_t, std::set<std::size_t>> adjacency;
        for (Finite_vertex_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
            Vertex_handle vh = vit;
            std::size_t vi = vertex_indices[vh];

            std::vector<Vertex_handle> neighbors;
            dt.finite_adjacent_vertices(vh, std::back_inserter(neighbors));

            for (Vertex_handle nh : neighbors) {
                adjacency[vi].insert(vertex_indices[nh]);
            }
            max_nb_adj = std::max(max_nb_adj, neighbors.size());
        }
        
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //std::cout << "Adjacencies computed in " << duration << endl;
        //std::cout << "Max nb adjacencies = " << max_nb_adj << endl;
        
        if (4*(int(max_nb_adj/4) + 1) > KVal_d) {
            fork_KVal_d = 4*(int(max_nb_adj/4) + 1);
            std::cout << "KVal = " << fork_KVal_d << endl;
            
            fork_ret_index.resize(fork_KVal_d*(NbPts+buff_symmetry));
        }
        
        unsigned int id = 0;
        for (const auto& [vi, neighbors] : adjacency) {
            int count = 0;
            for (std::size_t ni : neighbors) {
                fork_ret_index[fork_KVal_d*id + count] = static_cast<unsigned int>(ni);
                count++;
            }
            for (int j = count; j < KVal_d; j++) {
                fork_ret_index[fork_KVal_d*id + j] = id;
            }
            id++;
        }

        delaunay_done = 1;

        /*for (int i = 0; i < pts.size(); i++) {
            std::cout << "Neighbors of point index " << i << ": ";
            for (int j = 0; j < KVal; j++) {
                std::cout << ret_index[KVal*i + j] << " ";
            }
            std::cout << "\n";
        }*/
        
        // Print adjacency list
        /*for (const auto& [vi, neighbors] : adjacency) {
            std::cout << "Neighbors of point index " << vi << ": ";
            for (std::size_t ni : neighbors) {
                std::cout << ni << " ";
            }
            std::cout << "\n";
        }*/
        
        
        /*std::map<Point, std::set<Point>> point_to_adjacent_points;
        size_t max_nb_adj = 0;
        
        for (Finite_vertex_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
            Vertex_handle vh = vit;
            Point current_point = vh->point();

            std::set<Point> neighbors;
            std::vector<Vertex_handle> incident_vertices;

            dt.finite_adjacent_vertices(vh, std::back_inserter(incident_vertices));

            for (Vertex_handle neighbor : incident_vertices) {
                neighbors.insert(neighbor->point());
            }

            point_to_adjacent_points[current_point] = neighbors;
            max_nb_adj = std::max(max_nb_adj, neighbors.size());
        }*/
        
        // Print result
        /*for (const auto& entry : point_to_adjacent_points) {
            const Point& p = entry.first;
            std::cout << "Neighbors of (" << p << "):\n";
            for (const Point& q : entry.second) {
                std::cout << "  --> (" << q << ")\n";
            }
        }*/
    }

    void update_after_delaunay() {
        auto start = std::chrono::high_resolution_clock::now();
        if (fork_KVal_d > KVal_d) {
            KVal_d = fork_KVal_d;

            cudaMalloc(&d_adjacencies_delaunay, KVal_d * pts.size() * sizeof(unsigned int));
            cudaMalloc(&d_adjacencies, (KVal + KVal_d) * pts.size() * sizeof(unsigned int));
            
            /*ret_index.resize((KVal + KVal_d) * (NbPts + buff_symmetry));
            out_dist_sqr.resize((KVal+KVal_d)*(NbPts+buff_symmetry));

            glGenBuffers(1, &tbo_adjacents);
            glBindBuffer(GL_TEXTURE_BUFFER, tbo_adjacents);
            glBufferData(GL_TEXTURE_BUFFER, (KVal+KVal_d) * pts.size() * sizeof(unsigned int), nullptr, GL_STATIC_DRAW);*/
        }
        //memcpy(ret_index.data(), fork_ret_index.data(), (KVal+KVal_d) * pts.size() * sizeof(unsigned int));

        cudaMemcpy(d_adjacencies_delaunay, fork_ret_index.data(), KVal_d * pts.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        delaunay_done = 0;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //std::cout << "update_after_delaunay computed in " << duration << endl;
    }

    void update_thresh() {
        float min_lvl_0 = 1.0e32f;
        float min_lvl_1 = 1.0e32f;
        int count = 0;
        for (const auto& point : pts) {
            float curr_sdf = h_SDF_func(point); //length(point) - 0.5f;
            if ((curr_sdf - thresh_vals[0]) > 0.0f && abs(curr_sdf - thresh_vals[0]) < min_lvl_0) {
                min_lvl_0 = abs(curr_sdf - thresh_vals[0]);
            }
            if ((curr_sdf - thresh_vals[1]) < 0.0f &&  abs(curr_sdf - thresh_vals[1]) < min_lvl_1) {
                min_lvl_1 = abs(curr_sdf - thresh_vals[1]);
            }
            count++;
            if (count > NbPts)
                break;
        }
            
        if (abs(thresh_vals[0]) > abs(thresh_vals[1]))
            thresh_vals[0] = fmin(thresh_vals[0] + fmax(0.000f, min_lvl_0)/4.0f, -0.01f); 
        else
            thresh_vals[1] = fmax(thresh_vals[1] - fmax(0.000f, min_lvl_1)/4.0f, 0.01f); 
        std::cout << "thresh_vals[0] " << thresh_vals[0] << " -- min_lvl_0 " << min_lvl_0 << std::endl;
        std::cout << "thresh_vals[1] " << thresh_vals[1] << " -- min_lvl_1 " << min_lvl_1 << std::endl;
    }

    void make_symmetry() {
        //std::cout << "make_symmetry " << std::endl;
        int id = 0;
        int count = 0;
        for (const auto& f : flags) {
            
            if (f == 0) {
                id++;
                if (id == NbPts)
                    break;
                continue;
            }
            //std::cout << "flags " << f <<  " cast " << int(f) - 1 << std::endl;
            
            float3 point = pts[id];
            float sdf_val = h_SDF_func(point); //std::abs(h_SDF_func(point) - thresh_vals[int(f) - 1]); //std::abs((length(point) - 0.5f) - thresh_vals[int(f) - 1]);
            float3 grad_sdf = h_SDF_Grad_func(point); //normalize(point); // !! this should be replaced with true SDFgradients.
            
            if (f == 1) {
                pts[NbPts + count] = (point - 2.0f * std::abs(sdf_val - thresh_vals[0])* grad_sdf);

                indices_pt[NbPts + count] = NbPts + count;
                for (int j = 0; j < 20; j++) {
                    indices[20 * (NbPts + count) + j] = 20 * (NbPts + count) + j;
                }
                count++;
            }
            else if (f == 2) {
                pts[NbPts + count] = (point + 2.0f * std::abs(sdf_val - thresh_vals[1]) * grad_sdf);

                indices_pt[NbPts + count] = NbPts + count;
                for (int j = 0; j < 20; j++) {
                    indices[20 * (NbPts + count) + j] = 20 * (NbPts + count) + j;
                }
                count++;
            }
            else if (f == 3) {
                pts[NbPts + count] = (point - 2.0f * std::abs(sdf_val - thresh_vals[0]) * grad_sdf);

                indices_pt[NbPts + count] = NbPts + count;
                for (int j = 0; j < 20; j++) {
                    indices[20 * (NbPts + count) + j] = 20 * (NbPts + count) + j;
                }
                count++;

                pts[NbPts + count] = (point + 2.0f * std::abs(sdf_val - thresh_vals[1]) * grad_sdf);

                indices_pt[NbPts + count] = NbPts + count;
                for (int j = 0; j < 20; j++) {
                    indices[20 * (NbPts + count) + j] = 20 * (NbPts + count) + j;
                }
                count++;
            }
            //std::cout << "point " << point.x << ", " << point.y << ", " << point.z << std::endl;
            //std::cout << "pts[NbPts + count] " << pts[NbPts + count].x << ", " << pts[NbPts + count].y << ", " << pts[NbPts + count].z << std::endl;
            //std::cout << "sdf_val " << (length(pts[NbPts + count]) - 0.5) << ", " << (length(point) - 0.5) << ", " << sdf_val << ", " << thresh_vals[int(f) - 1] << std::endl;
            //std::cout << "grad_sdf " << grad_sdf.x << ", " << grad_sdf.y << ", " << grad_sdf.z << std::endl;

            
            id++;
            if (id == NbPts || count >= buff_symmetry-1)
                break;
        }

        CurrNbPts = NbPts + count;

        glBindBuffer(GL_TEXTURE_BUFFER, tbo);
        void* srcPtr = glMapBuffer(GL_TEXTURE_BUFFER, GL_WRITE_ONLY);
        memcpy(static_cast<char*>(srcPtr) + NbPts * sizeof(float3), reinterpret_cast<const char*>(pts.data()) + NbPts * sizeof(float3), count * sizeof(float3));
        glUnmapBuffer(GL_TEXTURE_BUFFER);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx);
        srcPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(static_cast<char*>(srcPtr) + 20*NbPts * sizeof(GLuint), reinterpret_cast<const char*>(indices.data()) + 20*NbPts * sizeof(GLuint), count * sizeof(GLuint));
        glUnmapBuffer(GL_ARRAY_BUFFER);


        glBindBuffer(GL_ARRAY_BUFFER, vbo_indx_pt);
        srcPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(static_cast<char*>(srcPtr) + NbPts * sizeof(GLuint), reinterpret_cast<const char*>(indices_pt.data()) + NbPts * sizeof(GLuint), count * sizeof(GLuint));
        glUnmapBuffer(GL_ARRAY_BUFFER);


        float3* dptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_vbo_in_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_in_resource);
        //std::cout << "num_bytes  " << num_bytes << std::endl;
        //std::cout << "size  " << (NbPts+count) * sizeof(float3) << std::endl;

        //cudaMemcpy(dptr + NbPts, pts.data() + NbPts, count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(dptr, pts.data(), (NbPts + count) * sizeof(float3), cudaMemcpyHostToDevice);

        cudaGraphicsUnmapResources(1, &cuda_vbo_in_resource, 0);
        
        //std::cout << "count " << count << std::endl;
        //std::cout << "NbPts " << NbPts << std::endl;
        //std::cout << "CurrNbPts " << CurrNbPts << std::endl;
    }

    void upsample() {
        pts.resize(NbPts);
        dt.clear();
        
        for (const auto& point : pts) {
            dt.insert(Point(point.x, point.y, point.z));
        }

        for (auto cell = dt.finite_cells_begin(); cell != dt.finite_cells_end(); ++cell) {
            // Each cell is a tetrahedron with 4 vertices
            Point p0 = cell->vertex(0)->point();
            Point p1 = cell->vertex(1)->point();
            Point p2 = cell->vertex(2)->point();
            Point p3 = cell->vertex(3)->point();

            float sdf_0 = h_SDF_func(make_float3(p0.x(), p0.y(), p0.z())); //(std::sqrt(CGAL::squared_distance(Point(0, 0, 0), p0)) - 0.5);
            float sdf_1 = h_SDF_func(make_float3(p1.x(), p1.y(), p1.z())); //(std::sqrt(CGAL::squared_distance(Point(0, 0, 0), p1))-0.5);
            float sdf_2 = h_SDF_func(make_float3(p2.x(), p2.y(), p2.z())); //(std::sqrt(CGAL::squared_distance(Point(0, 0, 0), p2))-0.5);
            float sdf_3 = h_SDF_func(make_float3(p3.x(), p3.y(), p3.z())); //(std::sqrt(CGAL::squared_distance(Point(0, 0, 0), p3))-0.5);
            if (((sdf_0-thresh_vals[0]) > 0.0f && (sdf_0-thresh_vals[1]) < 0.0f) &&
                ((sdf_1-thresh_vals[0]) > 0.0f && (sdf_1-thresh_vals[1]) < 0.0f) &&
                ((sdf_2-thresh_vals[0]) > 0.0f && (sdf_2-thresh_vals[1]) < 0.0f) &&
                ((sdf_3-thresh_vals[0]) > 0.0f && (sdf_3-thresh_vals[1]) < 0.0f)) {
                pts.push_back(make_float3((p0.x()+p1.x()+p2.x()+p3.x())/4.0f,
                                        (p0.y()+p1.y()+p2.y()+p3.y())/4.0f,
                                        (p0.z()+p1.z()+p2.z()+p3.z())/4.0f));

                indices_pt.push_back(pts.size() - 1);
                for (int j = 0; j < 20; j++) {
                    indices.push_back(20*(pts.size() - 1) + j);
                }
                //std::cout << "add point" << std::endl;
            }
        }
    

        NbPts = pts.size();
        CurrNbPts = NbPts;
        
        if (NbPts <= 1000) {
            buff_symmetry = 2000;
        } else if (NbPts <= 10000) {
            buff_symmetry = 10000;
        } else if (NbPts <= 100000) {
            buff_symmetry = 100000;
        } else if (NbPts <= 1000000) {
            buff_symmetry = 500000;
        }

        for (int i = 0; i < buff_symmetry; i++) {
            pts.push_back(make_float3(0.0f, 0.0f, 0.0f));
            indices_pt.push_back(0);
            for (int j = 0; j < 20; j++) {
                indices.push_back(0);
            }
        }

        //indices.resize(NbPts+buff_symmetry);
        fork_pts.resize((NbPts+buff_symmetry));
        float_pts.resize(NbPts+buff_symmetry);
        rgba.resize(4*(NbPts+buff_symmetry));
        flags.resize(NbPts+buff_symmetry);
        ret_index.resize((KVal+KVal_d)*(NbPts+buff_symmetry));
        fork_ret_index.resize(KVal_d * (NbPts + buff_symmetry));
        out_dist_sqr.resize((KVal+KVal_d)*(NbPts+buff_symmetry));
    }

    void Render3DGS(glm::mat4 projection, glm::mat4 view, glm::mat4 model_pose, GUI2D* gui_2D,
        int width_tex, int height_tex, int width, int height) {
        /*************** STEP 1. SORT points in ascending order w.r.t distance from camera ************** */
        // 1. Extract camera position from inverse of view matrix
        glm::vec3 camera_pos = glm::vec3(glm::inverse(view)[3]);
        //glm::vec3 camera_pos = glm::vec3(view[3]);

        // 2. Sort indices by comparing the depth of the corresponding pts
        std::sort(indices_pt.begin(), indices_pt.end(),
            [&](size_t i1, size_t i2) {
                float3 pt = pts[i1];
                float d1 = glm::length(glm::vec3(pt.x, pt.y, pt.z) - camera_pos); //(view * glm::vec4(pt.x, pt.y, pt.z, 1.0f)).z; //glm::length(pts[i1] - camera_pos);
                pt = pts[i2];
                float d2 = glm::length(glm::vec3(pt.x, pt.y, pt.z) - camera_pos); //(view * glm::vec4(pt.x, pt.y, pt.z, 1.0f)).z; //glm::length(pts[i2] - camera_pos);
                //float d1 = glm::length(pts[i1] - camera_pos);
                //float d2 = glm::length(pts[i2] - camera_pos);
                return d1 > d2; // far to near
            });

        // TEST
        for (size_t i = 0; i < pts.size(); ++i) {
            f_dc[i] = glm::vec3(abs(pts[i].x), abs(pts[i].y), abs(pts[i].z));
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo_dc);
        glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(glm::vec3), f_dc.data(), GL_STATIC_DRAW);
        //TEST

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, pts.size() * sizeof(unsigned int), indices_pt.data(), GL_DYNAMIC_DRAW);

        /*std::cout << "Projection" << endl;
        std::cout << projection[0].x << ", " << projection[0].y << ", " << projection[0].z << ", " << endl;
        std::cout << projection[1].x << ", " << projection[1].y << ", " << projection[1].z << ", " << endl;
        std::cout << projection[2].x << ", " << projection[2].y << ", " << projection[2].z << ", " << endl;
        */

        /*************** STEP2. Render in frame buffer object with local transmittance *********************** */
        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0, 0, width_tex, height_tex);

        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //glBlendFunc(GL_ONE, GL_ONE);
        //glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
        glDisable(GL_DEPTH_TEST);    // Disable depth testing!
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        ///// RENDER CVT
        glUseProgram(shader_3DGS_programme);

        glm::mat4 projection_FBO = glm::perspective(50.0, double(width_tex) / double(height_tex), 0.1, 30.0);
        GLint myLoc = glGetUniformLocation(shader_3DGS_programme, "projection");
        glUniformMatrix4fv(myLoc, 1, GL_FALSE, glm::value_ptr(projection_FBO));
        GLint viewLoc = glGetUniformLocation(shader_3DGS_programme, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        glUniform1f(glGetUniformLocation(shader_3DGS_programme, "GSize"), gui_2D->GauSize);
        glUniform1f(glGetUniformLocation(shader_3DGS_programme, "GScale"), gui_2D->pointSize);

        glBindVertexArray(vao_GS);
        glDrawElements(GL_POINTS, CurrNbPts, GL_UNSIGNED_INT, 0);
        //glDrawArrays(GL_POINTS, 0, pts.size());
        glBindVertexArray(0);

        //////

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);


        /*************** STEP3. Accumulate transmittance with order and render image on screen *********************** */
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // render to screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, width, height);

        glm::mat4 idmat = glm::mat4(1.0f);

        glUseProgram(shader_programme_normalize);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accumTex);
        glUniform1i(glGetUniformLocation(shader_programme_normalize, "accumTex"), 0);
        myLoc = glGetUniformLocation(shader_programme_normalize, "projection");
        //glUniformMatrix4fv(myLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(myLoc, 1, GL_FALSE, glm::value_ptr(idmat));
        viewLoc = glGetUniformLocation(shader_programme_normalize, "view");
        //glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(idmat));

        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 30 * sizeof(float), quadVertices);

        glDrawArrays(GL_TRIANGLES, 0, 6);

    }

    void RenderGS_Briac(glm::mat4 projection, glm::mat4 view, glm::mat4 model_pose, GUI2D* gui_2D,
        int width_tex, int height_tex, int width, int height) {
        // sort the gaussians by depth
        {
            auto& q = timers[OPERATIONS::SORT].push_back();
            q.begin();
            //sort.sort(gaussians_depths, sorted_depths, gaussians_indices, sorted_gaussian_indices, num_visible_gaussians);
            q.end();
        }
    }
};

#endif
