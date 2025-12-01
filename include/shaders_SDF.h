#ifndef __SHADERS_SDF_H
#define __SHADERS_SDF_H

#pragma once

const char* vertexShaderSDF_X = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
//"   vec4 Pos = vec4((aTexCoord.x*size[0]-K_c[0][1])/K_c[0][0], (aTexCoord.y*size[1]-K_c[1][2])/K_c[1][0], (layer.x*size[2]-K_c[2][1])/K_c[2][0], 1.0);\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*layer.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*aTexCoord.x, K_c[2][0] + (K_c[2][1]-K_c[2][0])*aTexCoord.y, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(aTexCoord.y, layer.x, aTexCoord.x);\n"
"}\n\0";


const char* vertexShaderSDF_Y = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
//"   vec4 Pos = vec4((aTexCoord.x*size[0]-K_c[0][1])/K_c[0][0], (aTexCoord.y*size[1]-K_c[1][2])/K_c[1][0], (layer.x*size[2]-K_c[2][1])/K_c[2][0], 1.0);\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*aTexCoord.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*layer.y, K_c[2][0] + (K_c[2][1]-K_c[2][0])*aTexCoord.y, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(aTexCoord.y, aTexCoord.x, layer.y);\n"
"}\n\0";


const char* vertexShaderSDF_Z = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
//"   vec4 Pos = vec4((aTexCoord.x*size[0]-K_c[0][1])/K_c[0][0], (aTexCoord.y*size[1]-K_c[1][2])/K_c[1][0], (layer.x*size[2]-K_c[2][1])/K_c[2][0], 1.0);\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*aTexCoord.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*aTexCoord.y, K_c[2][0] + (K_c[2][1]-K_c[2][0])*layer.z, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(layer.z, aTexCoord.x, aTexCoord.y);\n"
"}\n\0";

const char* fragmentShaderSDF = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 outTexCoord;\n"
"uniform sampler3D sdfTexture;\n"
"uniform sampler1D colormappingTexture;\n"
"uniform vec2 mag;\n"
"void main()\n"
"{\n"
"   float sdf = mag.x*(texture3D(sdfTexture, outTexCoord).r - 0.5f - mag.y);\n"
//"   sdf = sdf < mag.x/4.0f ? 0.0: (sdf-mag.x/4.0f) / (mag.x/2.0f);\n"
"   sdf = (1.0f+sdf)/2.0f;\n"
"   sdf = sdf < 0.0 ? 0.0: sdf;\n"
"   sdf = sdf > 1.0 ? 1.0: sdf;\n"
//"   sdf = mod(sdf, 0.01) < 0.001 ? 0.0: sdf;\n"
"   FragColor = texture1D(colormappingTexture, sdf);\n"
//"   FragColor = vec4(vec3(gl_FragCoord.z), 1.0);\n"
//"   FragColor = vec4(sdf, sdf, sdf, 1.0);\n"
"}\n\0";


void buildShaderSDF(GLuint *shader_programme_sdf_x, GLuint* shader_programme_sdf_y, GLuint* shader_programme_sdf_z) {
    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader_x = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader_x, 1, &vertexShaderSDF_X, NULL);
    glCompileShader(vertexShader_x);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader_x, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader_x, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }


    unsigned int vertexShader_y = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader_y, 1, &vertexShaderSDF_Y, NULL);
    glCompileShader(vertexShader_y);
    // check for shader compile errors
    glGetShaderiv(vertexShader_y, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader_y, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int vertexShader_z = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader_z, 1, &vertexShaderSDF_Z, NULL);
    glCompileShader(vertexShader_z);
    // check for shader compile errors
    glGetShaderiv(vertexShader_z, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader_z, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }


    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSDF, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    *shader_programme_sdf_x = glCreateProgram();
    glAttachShader(*shader_programme_sdf_x, vertexShader_x);
    glAttachShader(*shader_programme_sdf_x, fragmentShader);
    glLinkProgram(*shader_programme_sdf_x);
    // check for linking errors
    glGetProgramiv(*shader_programme_sdf_x, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_sdf_x, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_x);

    glUseProgram(*shader_programme_sdf_x);
    GLint sdfTexLocation = glGetUniformLocation(*shader_programme_sdf_x, "sdfTexture");
    GLint colormappingTexLocation = glGetUniformLocation(*shader_programme_sdf_x, "colormappingTexture");

    glUniform1i(sdfTexLocation, 0);
    glUniform1i(colormappingTexLocation, 1);

    // Y cut

    *shader_programme_sdf_y = glCreateProgram();
    glAttachShader(*shader_programme_sdf_y, vertexShader_y);
    glAttachShader(*shader_programme_sdf_y, fragmentShader);
    glLinkProgram(*shader_programme_sdf_y);
    // check for linking errors
    glGetProgramiv(*shader_programme_sdf_y, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_sdf_y, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_y);

    glUseProgram(*shader_programme_sdf_y);
    sdfTexLocation = glGetUniformLocation(*shader_programme_sdf_y, "sdfTexture");
    colormappingTexLocation = glGetUniformLocation(*shader_programme_sdf_y, "colormappingTexture");

    glUniform1i(sdfTexLocation, 0);
    glUniform1i(colormappingTexLocation, 1);

    // Z cut

    *shader_programme_sdf_z = glCreateProgram();
    glAttachShader(*shader_programme_sdf_z, vertexShader_z);
    glAttachShader(*shader_programme_sdf_z, fragmentShader);
    glLinkProgram(*shader_programme_sdf_z);
    // check for linking errors
    glGetProgramiv(*shader_programme_sdf_z, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_sdf_z, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_z);
    glDeleteShader(fragmentShader);

    glUseProgram(*shader_programme_sdf_z);
    sdfTexLocation = glGetUniformLocation(*shader_programme_sdf_z, "sdfTexture");
    colormappingTexLocation = glGetUniformLocation(*shader_programme_sdf_z, "colormappingTexture");

    glUniform1i(sdfTexLocation, 0);
    glUniform1i(colormappingTexLocation, 1);
}


//rendering pointcloud
void renderSDF_X(GLuint shader_programme_sdf_x, GLuint *sdf_texture, GLuint* color_mapping_texture, GLuint VertexArrayDepth, GLuint ElemArrayDepth,
                        GLfloat *mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    //layer.x = float((int(layer.x*512.0f) + 1) % 512)/512.0f;
    GLint layer_id = glGetUniformLocation(shader_programme_sdf_x, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_sdf_x, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));

    GLint mag_id = glGetUniformLocation(shader_programme_sdf_x, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, sdf_texture[0]);

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_1D, color_mapping_texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArrayAPPLE(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

//rendering pointcloud
void renderSDF_Y(GLuint shader_programme_sdf_y, GLuint* sdf_texture, GLuint* color_mapping_texture, GLuint VertexArrayDepth, GLuint ElemArrayDepth,
    GLfloat* mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    //Y cut
    GLint layer_id = glGetUniformLocation(shader_programme_sdf_y, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_sdf_y, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));
    GLint mag_id = glGetUniformLocation(shader_programme_sdf_y, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, sdf_texture[0]);

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_1D, color_mapping_texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArray(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

//rendering pointcloud
void renderSDF_Z(GLuint shader_programme_sdf_z, GLuint* sdf_texture, GLuint* color_mapping_texture, GLuint VertexArrayDepth, GLuint ElemArrayDepth,
    GLfloat* mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    // Z cut
    GLint layer_id = glGetUniformLocation(shader_programme_sdf_z, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_sdf_z, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));
    GLint mag_id = glGetUniformLocation(shader_programme_sdf_z, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, sdf_texture[0]);

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_1D, color_mapping_texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArray(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}


#endif
