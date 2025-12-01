#ifndef __UTILITIES_H
#define __UTILITIES_H



#pragma once


#ifdef _APPLE_
    //#define GLFW_INCLUDE_GLCOREARB
    //#include <OpenGL/OpenGL.h>
    //#include <OpenGL/gl.h>
    //#include <OpenGL/glu.h>
    //#include <OpenGL/gl3.h>
    //#include <OpenGL/glu.h>
    //#include <OpenGL/glext.h>
    //#include <GLFW/glfw3.h>
    //#include <GLUT/glut.h>
#else
    
   
   // #include <GL/glut.h>
   
    
#endif


/*** Include files for OpenGL to work ***/
#include <glad/gl.h>
//#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui/imgui.h"
//#define GLFW_INCLUDE_GLCOREARB
//#include <backends/imgui_impl_glfw.h>
//#include <backends/imgui_impl_opengl3.h>

/*** Standard include files for manipulating vectors, files etc... ***/
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <chrono>
#include <random>
#include <functional>
#include <future>
#include <thread>
#include <cstring>

#ifdef _APPLE_
    #include <sys/time.h>
#else
    #include <time.h>
    #include <Windows.h>
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
//#include <opencv/cv.hpp>

/*** Include files to manipulate matrices ***/
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Sparse>

#include "tiny_obj_loader.h"
#include <cuda_runtime.h>

#define DISPLAY_FRAME_IN true

using namespace std;

#ifdef _APPLE_
#define PATH_DATA string("/Users/diegothomas/Documents/Projects/Data/")
#else
#define PATH_DATA string("D:/Data/TetraModel/")
#endif

#undef HAVE_UNISTD_H


float h_SDF_Sphere(float3 point, float R) {
    return sqrt(point.x * point.x + point.y * point.y + point.z * point.z) - R;
}

float h_SDF_Torus(float3 point, float R_max, float R_min) {
    float qx = sqrt(point.x * point.x + point.y * point.y) - R_max;
    float qz = point.z;
    return sqrt(qx * qx + qz * qz) - R_min;
}

float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float h_SDF_Capsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);
    float3 ba = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float h = fmaxf(0.0f, fminf(1.0f, dot3(pa, ba) / dot3(ba, ba)));
    float3 fp = make_float3(pa.x - ba.x * h, pa.y - ba.y * h, pa.z - ba.z * h); //pa - ba * h;
    return sqrt(fp.x * fp.x + fp.y * fp.y + fp.z * fp.z) - r; //length(pa - ba * h) - r;
}

// Helper to rotate (y, z) vector
void h_rotate_yz(float* y, float* z, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    float newY = c * (*y) - s * (*z);
    float newZ = s * (*y) + c * (*z);
    *y = newY;
    *z = newZ;
}

float h_SDF_Bean(float3 p) {
    // Define bean spine
    float3 a = make_float3(-0.5, 0.0, 0.0);
    float3 b = make_float3(0.5, 0.0, 0.0);

    // Warp space for asymmetry (bend in Y-Z plane)
    float bend = 5.0f;
    float theta = bend * p.x; // bending angle
    h_rotate_yz(&p.y, &p.z, theta);

    return h_SDF_Capsule(p, a, b, 0.3f);
}

float h_SDF_func(float3 point) {
    //return h_SDF_Sphere(point, 0.5f);
    return h_SDF_Torus(point, 0.6f, 0.4f);
    //return h_SDF_Bean(point);
}

float3 h_SDF_Grad_Sphere(float3 point) {
    float norm = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    return norm == 0.0f ? point : make_float3(point.x / norm, point.y / norm, point.z / norm);
}

float3 h_SDF_Grad_Torus(float3 point, float R_max, float R_min) {
    float a = sqrt(point.x * point.x + point.y * point.y);
    float b = sqrt((a - R_max) * (a - R_max) + point.z * point.z);

    return make_float3(((a - R_max) / b) * (point.x / a), ((a - R_max) / b) * (point.y / a), point.z / b);
}

float3 h_normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-8f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    else
        return make_float3(0.0f, 0.0f, 0.0f); // Avoid division by zero
}

float3 h_SDF_Gradient_FD(float3 p) {
    const float eps = 1e-4f; // Small step
    return h_normalize(make_float3(
        h_SDF_func(make_float3(p.x + eps, p.y, p.z)) - h_SDF_func(make_float3(p.x - eps, p.y, p.z)),
        h_SDF_func(make_float3(p.x, p.y + eps, p.z)) - h_SDF_func(make_float3(p.x, p.y - eps, p.z)),
        h_SDF_func(make_float3(p.x, p.y, p.z + eps)) - h_SDF_func(make_float3(p.x, p.y, p.z - eps))
    ));
}

float3 h_SDF_Grad_func(float3 point) {
    //return h_SDF_Grad_Sphere(point);
    return h_SDF_Grad_Torus(point, 0.6f, 0.4f);
    //return h_SDF_Gradient_FD(point);
}

static void PrintInfo(const tinyobj::attrib_t& attrib,
    const std::vector<tinyobj::shape_t>& shapes,
    const std::vector<tinyobj::material_t>& materials) {
    std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
    std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
    std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2)
        << std::endl;

    std::cout << "# of shapes    : " << shapes.size() << std::endl;
    std::cout << "# of materials : " << materials.size() << std::endl;

    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.vertices[3 * v + 0]),
            static_cast<const double>(attrib.vertices[3 * v + 1]),
            static_cast<const double>(attrib.vertices[3 * v + 2]));
    }

    for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
        printf("  n[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.normals[3 * v + 0]),
            static_cast<const double>(attrib.normals[3 * v + 1]),
            static_cast<const double>(attrib.normals[3 * v + 2]));
    }

    for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
        printf("  uv[%ld] = (%f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.texcoords[2 * v + 0]),
            static_cast<const double>(attrib.texcoords[2 * v + 1]));
    }

    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        printf("shape[%ld].name = %s\n", static_cast<long>(i),
            shapes[i].name.c_str());
        printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.indices.size()));
        printf("Size of shape[%ld].lines.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].lines.indices.size()));
        printf("Size of shape[%ld].points.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].points.indices.size()));

        size_t index_offset = 0;

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.material_ids.size());

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.smoothing_group_ids.size());

        printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
                static_cast<unsigned long>(fnum));

            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                printf("    face[%ld].v[%ld].idx = %d/%d/%d\n", static_cast<long>(f),
                    static_cast<long>(v), idx.vertex_index, idx.normal_index,
                    idx.texcoord_index);
            }

            printf("  face[%ld].material_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.material_ids[f]);
            printf("  face[%ld].smoothing_group_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.smoothing_group_ids[f]);

            index_offset += fnum;
        }

        printf("shape[%ld].num_tags: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.tags.size()));
        for (size_t t = 0; t < shapes[i].mesh.tags.size(); t++) {
            printf("  tag[%ld] = %s ", static_cast<long>(t),
                shapes[i].mesh.tags[t].name.c_str());
            printf(" ints: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].intValues.size(); ++j) {
                printf("%ld", static_cast<long>(shapes[i].mesh.tags[t].intValues[j]));
                if (j < (shapes[i].mesh.tags[t].intValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" floats: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].floatValues.size(); ++j) {
                printf("%f", static_cast<const double>(
                    shapes[i].mesh.tags[t].floatValues[j]));
                if (j < (shapes[i].mesh.tags[t].floatValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" strings: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].stringValues.size(); ++j) {
                printf("%s", shapes[i].mesh.tags[t].stringValues[j].c_str());
                if (j < (shapes[i].mesh.tags[t].stringValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");
            printf("\n");
        }
    }

    for (size_t i = 0; i < materials.size(); i++) {
        printf("material[%ld].name = %s\n", static_cast<long>(i),
            materials[i].name.c_str());
        printf("  material.Ka = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].ambient[0]),
            static_cast<const double>(materials[i].ambient[1]),
            static_cast<const double>(materials[i].ambient[2]));
        printf("  material.Kd = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].diffuse[0]),
            static_cast<const double>(materials[i].diffuse[1]),
            static_cast<const double>(materials[i].diffuse[2]));
        printf("  material.Ks = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].specular[0]),
            static_cast<const double>(materials[i].specular[1]),
            static_cast<const double>(materials[i].specular[2]));
        printf("  material.Tr = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].transmittance[0]),
            static_cast<const double>(materials[i].transmittance[1]),
            static_cast<const double>(materials[i].transmittance[2]));
        printf("  material.Ke = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].emission[0]),
            static_cast<const double>(materials[i].emission[1]),
            static_cast<const double>(materials[i].emission[2]));
        printf("  material.Ns = %f\n",
            static_cast<const double>(materials[i].shininess));
        printf("  material.Ni = %f\n", static_cast<const double>(materials[i].ior));
        printf("  material.dissolve = %f\n",
            static_cast<const double>(materials[i].dissolve));
        printf("  material.illum = %d\n", materials[i].illum);
        printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
        printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
        printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
        printf("  material.map_Ns = %s\n",
            materials[i].specular_highlight_texname.c_str());
        printf("  material.map_bump = %s\n", materials[i].bump_texname.c_str());
        printf("    bump_multiplier = %f\n", static_cast<const double>(materials[i].bump_texopt.bump_multiplier));
        printf("  material.map_d = %s\n", materials[i].alpha_texname.c_str());
        printf("  material.disp = %s\n", materials[i].displacement_texname.c_str());
        printf("  <<PBR>>\n");
        printf("  material.Pr     = %f\n", static_cast<const double>(materials[i].roughness));
        printf("  material.Pm     = %f\n", static_cast<const double>(materials[i].metallic));
        printf("  material.Ps     = %f\n", static_cast<const double>(materials[i].sheen));
        printf("  material.Pc     = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
        printf("  material.Pcr    = %f\n", static_cast<const double>(materials[i].clearcoat_roughness));
        printf("  material.aniso  = %f\n", static_cast<const double>(materials[i].anisotropy));
        printf("  material.anisor = %f\n", static_cast<const double>(materials[i].anisotropy_rotation));
        printf("  material.map_Ke = %s\n", materials[i].emissive_texname.c_str());
        printf("  material.map_Pr = %s\n", materials[i].roughness_texname.c_str());
        printf("  material.map_Pm = %s\n", materials[i].metallic_texname.c_str());
        printf("  material.map_Ps = %s\n", materials[i].sheen_texname.c_str());
        printf("  material.norm   = %s\n", materials[i].normal_texname.c_str());
        std::map<std::string, std::string>::const_iterator it(
            materials[i].unknown_parameter.begin());
        std::map<std::string, std::string>::const_iterator itEnd(
            materials[i].unknown_parameter.end());

        for (; it != itEnd; it++) {
            printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
        }
        printf("\n");
    }
}

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

std::string loadShaderSource(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open shader file: " << filePath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compileShader(GLenum type, const std::string& source) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    // Error checking
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }

    return shader;
}

#endif
