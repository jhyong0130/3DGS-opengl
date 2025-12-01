//
// Created by Briac on 18/06/2025.
//

#ifndef SPARSEVOXRECON_CAMERA_H
#define SPARSEVOXRECON_CAMERA_H

#include <iostream>
#include "../glad/gl.h"
#include "GLFW/glfw3.h"

#include "../glm/vec2.hpp"
#include "../glm/vec3.hpp"
#include "../glm/vec4.hpp"
#include "../glm/mat4x4.hpp"
#include "../glm/gtc/constants.hpp"   // for glm::pi
#include "../glm/gtc/matrix_transform.hpp" // for glm::degrees / radians

#include <numbers>

class Camera {
public:
    Camera();
    virtual ~Camera();

    void updateView(GLFWwindow *window, bool windowHovered, float scroll);

    void updateViewMatrix() {
        invViewMat = glm::inverse(viewMat);
        projViewMat = projMat * viewMat;
        invProjViewMat = glm::inverse(projViewMat);

        glm::mat3 R = glm::mat3(viewMat);         // upper-left 3x3 rotation
        glm::vec3 t = glm::vec3(viewMat[3]);      // translation column (4th column)

        // Camera position = - R^T * t
        camPos = -glm::transpose(R) * t;

    }

    void setPosition(const glm::vec3& pos) {
        camPos = pos;
        updateViewMatrix();
    }

    void setViewMatrix(const glm::mat4& view) {
        viewMat = view;
        updateViewMatrix();
    }

    void setProjectionMatrix(const glm::mat4& K) {
        projMat = glm::mat4(0.0f);
        projMat[0][0] = 2.0f * K[0][0] / float(framebufferSize.x);
        projMat[1][1] = 2.0f * K[1][1] / float(framebufferSize.y);

        projMat[2][0] = 1.0f - 2.0f * K[0][2] / float(framebufferSize.x);
        projMat[2][1] = 2.0f * K[1][2] / float(framebufferSize.y) - 1.0f;
        projMat[2][2] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        projMat[2][3] = -1.0f;

        projMat[3][2] = -2.0f * farPlane * nearPlane / (farPlane - nearPlane);

        invProj = glm::inverse(projMat);

        // fx and fy are usually at K[0][0] and K[1][1]
        float fx = K[0][0];
        float fy = K[1][1];
        //std::cout << K[0][0] << ", " << K[0][1] << ", " << K[0][2] << ", " << K[0][3] << std::endl;

        fovX = 2.0f * atan(framebufferSize.x / (2.0f * fx));
        fovY = 2.0f * atan(framebufferSize.y / (2.0f * fy));
    }

    void setFrameBufferSize(int width, int height);

    void copyViewMat2cuda();

    glm::vec3 getPosition() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getProjectionViewMatrix() const;

    glm::vec2 getMouseFramebufferCoords() const;
    glm::ivec2 getFramebufferSize() const;

    float getNearPlane() const;
    float& getFarPlane();

    float getFovX() const;
    float getFovY() const;

    float* viewMat_cu;
    float* camPos_cu;

private:
    float dist2lookPos = 3;
    float theta = 0;
    float phi = 0;

    float nearPlane = 0.001f;
    float farPlane = 100.0f;

    float fovX = 0.0f;
    float fovY = (float) std::numbers::pi / 4.0f;
    float camSpeed = 0.25f / 60.0f;

    glm::mat4 projMat;
    glm::mat4 viewMat;

    glm::mat4 invProj;
    glm::mat4 invViewMat;

    glm::mat4 projViewMat;
    glm::mat4 invProjViewMat;

    glm::vec3 camPos;
    glm::vec3 lookPos;
    glm::vec3 camDir;
    glm::vec3 camRight;
    glm::vec3 camUp;

    bool freeCam = false;
    bool movementsEnabled = true; // set to false in 2D mode

    bool ctrlKey;
    bool altKey;
    bool shiftKey;

    glm::ivec2 framebufferSize;
    glm::vec2 mouseFramebufferCoords;
    glm::vec2 mouseNormalizedCoords;
};


#endif //SPARSEVOXRECON_CAMERA_H
