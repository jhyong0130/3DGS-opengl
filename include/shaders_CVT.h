#ifndef __SHADERS_SDF_H
#define __SHADERS_SDF_H

#pragma once


std::pair<GLuint, GLuint> BuildShadersCVT() {
    GLuint v_shader = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("CVTsplat.vert")));
    GLuint g_shader = compileShader(GL_GEOMETRY_SHADER, loadShaderSource(string(SHADERS_PATH) + string("CVTsplat.geo")));
    GLuint f_shader = compileShader(GL_FRAGMENT_SHADER, loadShaderSource(string(SHADERS_PATH) + string("CVTsplat.frag")));
    GLuint v_shader_normalise = compileShader(GL_VERTEX_SHADER, loadShaderSource(string(SHADERS_PATH) + string("SimpleVertexShader.glsl")));
    GLuint f_shader_normalise = compileShader(GL_FRAGMENT_SHADER, loadShaderSource(string(SHADERS_PATH) + string("Normalise.glsl")));

     // link shaders
    GLuint shader_programme = glCreateProgram();
    glAttachShader(shader_programme, v_shader);
    glAttachShader(shader_programme, g_shader);
    glAttachShader(shader_programme, f_shader);
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
    glDeleteShader(g_shader);
    glDeleteShader(f_shader);

    GLuint shader_normalise = glCreateProgram();
    glAttachShader(shader_normalise, v_shader_normalise);
    glAttachShader(shader_normalise, f_shader_normalise);
    glLinkProgram(shader_normalise);
    // check for linking errors
    glGetProgramiv(shader_normalise, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_normalise, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::shader_normalise::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(v_shader_normalise);
    glDeleteShader(f_shader_normalise);

    return std::pair<GLuint, GLuint>(shader_programme, shader_normalise);
}

//rendering pointcloud
void renderCVT(GLuint shader_programme, GLuint accumFBO, GLuint VertexArray, int nbPoints, float GSize, float Scale,
    glm::vec3 p_center, glm::vec3 p_normal, glm::vec3 p_u, glm::vec3 p_v, 
    glm::mat4 projection, glm::mat4 view, float* cam_pose, int width, int height) {

    glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);

    glEnable(GL_BLEND);
    glBlendEquation(GL_MIN);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);    // Disable depth testing!
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    ///// RENDER CVT
    glUseProgram(shader_programme);

    // Plane cut
    GLint center_id = glGetUniformLocation(shader_programme, "p_center");
    glUniform3fv(center_id, 1, glm::value_ptr(p_center));
    GLint normal_id = glGetUniformLocation(shader_programme, "p_normal");
    glUniform3fv(normal_id, 1, glm::value_ptr(p_normal));
    
    GLint myLoc = glGetUniformLocation(shader_programme, "projection");
    glUniformMatrix4fv(myLoc, 1, GL_FALSE, glm::value_ptr(projection));
    GLint viewLoc = glGetUniformLocation(shader_programme, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    GLint modelLoc = glGetUniformLocation(shader_programme, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, cam_pose);
    GLint u_id = glGetUniformLocation(shader_programme, "p_u");
    glUniform3fv(u_id, 1, glm::value_ptr(p_u));
    GLint v_id = glGetUniformLocation(shader_programme, "p_v");
    glUniform3fv(v_id, 1, glm::value_ptr(p_v));

    glUniform1f(glGetUniformLocation(shader_programme, "GSize"), GSize);
    glUniform1f(glGetUniformLocation(shader_programme, "Scale"), Scale);

    glBindVertexArray(VertexArray);
    glDrawArrays(GL_POINTS, 0, nbPoints);

    //////

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST); 
}

void renderScreen(GLuint shader_programme_normalize, GLuint accumTex, 
    GLuint quadVAO, GLuint quadVBO, float* quadVertices, glm::mat4 projection, glm::mat4 view) {

    glUseProgram(shader_programme_normalize);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, accumTex);
    glUniform1i(glGetUniformLocation(shader_programme_normalize, "accumTex"), 0);
    GLint myLoc = glGetUniformLocation(shader_programme_normalize, "projection");
    glUniformMatrix4fv(myLoc, 1, GL_FALSE, glm::value_ptr(projection));
    GLint viewLoc = glGetUniformLocation(shader_programme_normalize, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 30*sizeof(float), quadVertices);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}

#endif
