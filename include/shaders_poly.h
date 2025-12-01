#ifndef __SHADERS_POLY_H
#define __SHADERS_POLY_H

#pragma once


const char* vertexShaderPoly = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aNormal;\n"
"out vec3 outColor;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"   gl_Position = projection * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   FragPos = vec3(model * vec4(aPos, 1.0));\n"
//"   outColor = aColor;\n"
"   outColor = vec3(1.0f, 1.0f, 1.0f);\n"
"   Normal = aNormal;\n"
"}\0";

const char* fragmentShaderPoly = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 outColor;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"uniform vec3 lightPos;\n"
"uniform vec3 lightColor;\n"
"void main()\n"
"{\n"
"   vec3 norm = normalize(Normal);\n"
"   vec3 lightDir = normalize(lightPos - FragPos);\n"
"   float diff = max(abs(dot(norm, lightDir)), 0.0);\n"
"   vec3 diffuse = diff * lightColor;\n"
"   float ambient = 0.1;\n"
"   vec3 result = (ambient + diffuse) * outColor;\n"
"   FragColor = vec4(result, 1.0);\n"
"}\n\0";

void buildShaderPoly(GLuint *shader_programme_poly) {
    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderPoly, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderPoly, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    *shader_programme_poly = glCreateProgram();
    glAttachShader(*shader_programme_poly, vertexShader);
    glAttachShader(*shader_programme_poly, fragmentShader);
    glLinkProgram(*shader_programme_poly);
    // check for linking errors
    glGetProgramiv(*shader_programme_poly, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shader_programme_poly, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}


#endif
