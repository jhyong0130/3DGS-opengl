#ifndef __SHADERS_GRAD_H
#define __SHADERS_GRAD_H

#pragma once


const char* vertexShaderGRAD_X = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*layer.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*aTexCoord.x, K_c[2][0] + (K_c[2][1]-K_c[2][0])*aTexCoord.y, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(aTexCoord.y, layer.x, aTexCoord.x);\n"
"}\n\0";


const char* vertexShaderGRAD_Y = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*aTexCoord.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*layer.y, K_c[2][0] + (K_c[2][1]-K_c[2][0])*aTexCoord.y, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(aTexCoord.y, aTexCoord.x, layer.y);\n"
"}\n\0";


const char* vertexShaderGRAD_Z = "#version 330 core\n"
"layout (location = 0) in vec2 aTexCoord;\n"
"out vec3 outTexCoord;\n"
"uniform vec3 layer;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"uniform mat3 K_c;\n"
"void main()\n"
"{\n"
"   vec4 Pos = vec4(K_c[0][0] + (K_c[0][1]-K_c[0][0])*aTexCoord.x, K_c[1][0] + (K_c[1][1]-K_c[1][0])*aTexCoord.y, K_c[2][0] + (K_c[2][1]-K_c[2][0])*layer.z, 1.0);\n"
"   gl_Position = projection * view * model * Pos;\n"
"   outTexCoord = vec3(layer.z, aTexCoord.x, aTexCoord.y);\n"
"}\n\0";

const char* fragmentShaderGRAD = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 outTexCoord;\n"
"uniform sampler3D Texture;\n"
"uniform vec2 mag;\n"
"void main()\n"
"{\n"
"   FragColor = mag.x*(texture3D(Texture, outTexCoord));\n"
"}\n\0";


void buildShaderGRAD(GLuint* shader_programme_x, GLuint* shader_programme_y, GLuint* shader_programme_z) {
    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader_x = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader_x, 1, &vertexShaderGRAD_X, NULL);
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
    glShaderSource(vertexShader_y, 1, &vertexShaderGRAD_Y, NULL);
    glCompileShader(vertexShader_y);
    // check for shader compile errors
    glGetShaderiv(vertexShader_y, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader_y, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int vertexShader_z = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader_z, 1, &vertexShaderGRAD_Z, NULL);
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
    glShaderSource(fragmentShader, 1, &fragmentShaderGRAD, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    *shader_programme_x = glCreateProgram();
    glAttachShader(*shader_programme_x, vertexShader_x);
    glAttachShader(*shader_programme_x, fragmentShader);
    glLinkProgram(*shader_programme_x);
    // check for linking errors
    glGetProgramiv(*shader_programme_x, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_x, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_x);

    glUseProgram(*shader_programme_x);
    GLint TexLocation = glGetUniformLocation(*shader_programme_x, "Texture");

    glUniform1i(TexLocation, 0);

    // Y cut

    *shader_programme_y = glCreateProgram();
    glAttachShader(*shader_programme_y, vertexShader_y);
    glAttachShader(*shader_programme_y, fragmentShader);
    glLinkProgram(*shader_programme_y);
    // check for linking errors
    glGetProgramiv(*shader_programme_y, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_y, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_y);

    glUseProgram(*shader_programme_y);
    TexLocation = glGetUniformLocation(*shader_programme_y, "Texture");

    glUniform1i(TexLocation, 0);

    // Z cut

    *shader_programme_z = glCreateProgram();
    glAttachShader(*shader_programme_z, vertexShader_z);
    glAttachShader(*shader_programme_z, fragmentShader);
    glLinkProgram(*shader_programme_z);
    // check for linking errors
    glGetProgramiv(*shader_programme_z, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_z, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader_z);
    glDeleteShader(fragmentShader);

    glUseProgram(*shader_programme_z);
    TexLocation = glGetUniformLocation(*shader_programme_z, "Texture");

    glUniform1i(TexLocation, 0);
}



//rendering pointcloud
void renderGRAD_X(GLuint shader_programme_x, GLuint* texture,  GLuint VertexArrayDepth, GLuint ElemArrayDepth,
    GLfloat* mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    //layer.x = float((int(layer.x*512.0f) + 1) % 512)/512.0f;
    GLint layer_id = glGetUniformLocation(shader_programme_x, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_x, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));

    GLint mag_id = glGetUniformLocation(shader_programme_x, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArray(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

//rendering pointcloud
void renderGRAD_Y(GLuint shader_programme_y, GLuint* texture, GLuint VertexArrayDepth, GLuint ElemArrayDepth,
    GLfloat* mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    //Y cut
    GLint layer_id = glGetUniformLocation(shader_programme_y, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_y, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));
    GLint mag_id = glGetUniformLocation(shader_programme_y, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArray(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

//rendering pointcloud
void renderGRAD_Z(GLuint shader_programme_z, GLuint* texture, GLuint VertexArrayDepth, GLuint ElemArrayDepth,
    GLfloat* mag_sdf, glm::vec3 layer, glm::mat3 cam1_intrinsics, glm::vec3 cam1_size) {
    // Z cut
    GLint layer_id = glGetUniformLocation(shader_programme_z, "layer");
    glUniform3fv(layer_id, 1, glm::value_ptr(layer));
    GLint depth_K = glGetUniformLocation(shader_programme_z, "K_c");
    glUniformMatrix3fv(depth_K, 1, GL_FALSE, glm::value_ptr(cam1_intrinsics));
    GLint mag_id = glGetUniformLocation(shader_programme_z, "mag");
    glUniform2fv(mag_id, 1, mag_sdf);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, texture[0]);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glBindVertexArray(VertexArrayDepth);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElemArrayDepth);
    glDrawElements(GL_TRIANGLES, 3 * 2 * int(cam1_size[0] - 1) * int(cam1_size[1] - 1), GL_UNSIGNED_INT, 0);
    //glDrawArrays(GL_POINTS, 0, int(cam1_size[0]) * int(cam1_size[1]));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

#endif
