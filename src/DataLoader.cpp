//
// Created by Briac on 27/08/2025.
//

#include "DataLoader.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include "glm/gtc/type_ptr.hpp" // <-- for glm::make_mat4


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


void DataLoader::load(const std::string& folder) {
    // Directory + filename pattern
    std::string prefix = "/image/"; //frame_
    std::string prefix_cam = "/pose/"; //frame_
    std::string extension = ".png";

    int currIndex = 0;   // first index
    int startIndex = 0;   // first index
    int endIndex = 10;  // last index
    int width = 3;   // number of digits in numbering (e.g., 001, 002)

    //for (int currIndex = startIndex; currIndex <= endIndex; currIndex++) {
    while(true) {
        std::ostringstream ss;
        ss << folder << prefix << std::setw(width) << std::setfill('0') << currIndex << extension;
        std::string filename = ss.str();

        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Could not load image: " << filename << std::endl;
            break;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGBA); // Ensure RGBA8

        unsigned char* img_gpu;
        cudaMalloc((void**)&img_gpu, img.cols * img.rows * 4 * sizeof(unsigned char));
        cudaMemcpy(img_gpu, img.data, img.cols * img.rows * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

        cols = img.cols;
        rows = img.rows;

        images_gpu.push_back(img_gpu);
        std::cout << "Loaded: " << filename << " (" << img.cols << "x" << img.rows << ")" << std::endl;

        // Load camera pose
        std::ostringstream ss_cam;
        ss_cam << folder << prefix_cam << std::setw(width) << std::setfill('0') << currIndex << ".txt";
        std::string cam_filename = ss_cam.str();
        std::ifstream file(cam_filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + cam_filename);
        }

        glm::mat4 mat(1.0f); // identity by default
        float values[16];
        int count = 0;

        while (file && count < 16) {
            file >> values[count];
            if (!file) break;
            count++;
        }

        if (count != 16) {
            throw std::runtime_error("File does not contain 16 floats: " + cam_filename);
        }

        // GLM is column-major, but your text file is row-major.
        // So we must transpose when assigning.
        mat = glm::make_mat4(values);
        mat = glm::transpose(mat);

        poses.push_back(mat);

        currIndex++;
    }

    std::cout << "Total images loaded: " << images_gpu.size() << std::endl;
    camera.setFrameBufferSize(cols, rows);

    // Example: show the first image
    /*if (!images.empty()) {
        cv::imshow("First image", images[0]);
        cv::waitKey(0);
    }*/

    // Load camera intrinsics parameters
    {
        std::ostringstream ss;
        ss << folder << "/intrinsics.txt";
        std::string cam_filename = ss.str();
        std::ifstream file(cam_filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + cam_filename);
        }

        glm::mat4 mat(1.0f); // identity by default
        float values[16];
        int count = 0;

        while (file && count < 16) {
            file >> values[count];
            if (!file) break;
            count++;
        }

        if (count != 16) {
            throw std::runtime_error("File does not contain 16 floats: " + cam_filename);
        }

        // GLM is column-major, but your text file is row-major.
        // So we must transpose when assigning.
        mat = glm::make_mat4(values);

        camera.setProjectionMatrix(mat);
    }


    return ;
}