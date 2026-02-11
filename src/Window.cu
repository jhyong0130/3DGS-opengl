//
// Created by Briac on 18/06/2025.
//

#include "Window.cuh"

#include <iostream>
#include <cstdint>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include "./RenderingBase/helper_cuda.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/ImGuiFileDialog.h"
#include "imgui/ImGuiFileDialogConfig.h"

#include "RenderingBase/Camera.h"
#include "RenderingBase/GLShaderLoader.h"
#include "RenderingBase/GLIntrospection.h"
#include "RenderingBase/CudaIntrospection.cuh"

#include "PointCloudLoader.h"
#include "DataLoader.h"

#include <thread>
#include <chrono>
#include <filesystem>

static void error_callback(int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
    fflush(stderr);
}

static void myGlDebugCallback(GLenum source,
                              GLenum type,
                              GLuint id,
                              GLenum severity,
                              GLsizei length,
                              const GLchar *message,
                              const void *userParam){

    if(severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM
    || severity == GL_DEBUG_SEVERITY_LOW){
        std::cout <<"GL_DEBUG: " <<message <<std::endl;
    }


}
static void glad_callback_custom(void *ret, const char *name, GLADapiproc apiproc, int len_args, ...) {
    // Guard against NULL function pointer (can be NULL if not loaded or context lost)
    if (glad_glGetError == NULL) {
        return;
    }

    GLenum error_code = glad_glGetError();
    if (error_code == GL_NO_ERROR) {
        return;
    }

    std::string type("UNKNOWN");
    if (error_code == GL_INVALID_ENUM) {
        type = "GL_INVALID_ENUM";
    } else if (error_code == GL_INVALID_OPERATION) {
        type = "GL_INVALID_OPERATION";
    } else if (error_code == GL_INVALID_VALUE) {
        type = "GL_INVALID_VALUE";
    } else if (error_code == GL_INVALID_INDEX) {
        type = "GL_INVALID_INDEX";
    } else if (error_code == GL_INVALID_FRAMEBUFFER_OPERATION) {
        type = "GL_INVALID_FRAMEBUFFER_OPERATION";
    } else if (error_code == GL_OUT_OF_MEMORY) {
        type = "GL_OUT_OF_MEMORY";
    } else if(error_code == GL_CONTEXT_LOST){
        type = "GL_CONTEXT_LOST";
    }

    std::cout << "ERROR " << error_code << " in " << name << " (" << type
              << ")" << std::endl;

    if (error_code == GL_OUT_OF_MEMORY) {
        throw std::string("OpenGL Fatal Error: Out of memory");
    }
}

static void framebuffer_size_callback(GLFWwindow *window, int width,
                                      int height) {
    glViewport(0, 0, width, height);
}

static double scroll;
static void scroll_callback(GLFWwindow *window, double xoffset,
                            double yoffset) {
    scroll = yoffset;
}

Window::Window(const std::string &title, int samples) {
    if (!glfwInit())
        throw "Error while initializing GLFW";

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_API, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, samples);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);
//    glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);
    glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);

    glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
//    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_FALSE);

//    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
//    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);

    w = glfwCreateWindow(800, 600, title.c_str(), NULL, NULL);
    if (!w) {
        glfwTerminate();
        throw std::string("Error while creating the window");
    }

    glfwMakeContextCurrent(w);
    gladLoadGL(glfwGetProcAddress);
    gladInstallGLDebug();

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
              << std::endl;

    glfwSetErrorCallback(error_callback);
    glfwSetFramebufferSizeCallback(w, framebuffer_size_callback);
    glfwSetScrollCallback(w, scroll_callback);

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // Only install the custom post-callback if glGetError is available
    if (glad_glGetError != NULL) {
        gladSetGLPostCallback(reinterpret_cast<GLADpostcallback>(glad_callback_custom));
    }
    glDebugMessageCallback(myGlDebugCallback, nullptr);

//    guis = std::make_unique<GUIs>();
//    guis->init_IMGUI(w);
//    reloadFonts();
//
//    setTitle();
//    toogleFullscreen();
//    toogleVsync();
    auto ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);
    ImGui_ImplOpenGL3_Init();
    ImGui_ImplGlfw_InitForOpenGL(w, true);
}

Window::~Window() {

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void loadHeaders(){

    std::function<void(std::unordered_map<std::string, std::string>&)> replacements;
    std::unordered_map<std::string, std::string> m;

    // glsl syntax hacks
    m["___flat"] = "flat";
    m["___out"] = "out";
    m["___in"] = "in";
    m["___inout"] = "inout";
    m["___discard"] = "discard";
    m["//--"] = "";
    m["/\\*--"] = "";
    m["--\\*/"] = "";

    m["#include"] = "";
    m["\\bstatic\\b"] = "";
    m["\\binline\\b"] = "";
    m["__UNKOWN_SIZE"] = "";


    std::string regex_str = "";
    for(const auto& [s, d] : m){
        if(regex_str.size() > 0){
            regex_str += "|" + s;
        }else{
            regex_str += s;
        }
    }
    std::regex re = std::regex(regex_str);

    std::vector<std::string> headers;
    headers.push_back("resources/shaders/common/GLSLDefines.h");
    headers.push_back("resources/shaders/common/CommonTypes.h");
    headers.push_back("resources/shaders/common/Uniforms.h");
    headers.push_back("resources/shaders/common/Covariance.h");
    GLShaderLoader::instance->loadHeaders(headers, m, re);
}

void Window::mainloop(int argc, char** argv) {

    Camera camera;
    GLShaderLoader shaderLoader("resources/shaders", "SparseVoxelReconstruction");
    loadHeaders();

    unsigned int gl_device_count;
    int gl_device_id;
    checkCudaErrors(cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));
    int cuda_device_id = gl_device_id;
    checkCudaErrors(cudaSetDevice(cuda_device_id));

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, gl_device_id));
    printf("GL   : %-24s (%2d SMs)\n", props.name, props.multiProcessorCount);
    checkCudaErrors(cudaGetDeviceProperties(&props, cuda_device_id));
    printf("CUDA : %-24s (%2d SMs)\n", props.name, props.multiProcessorCount);

    // Check available memory
    size_t free_mem, total_mem;
    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory: %.2f MB free / %.2f MB total\n",
        free_mem / (1024.0f * 1024.0f),
        total_mem / (1024.0f * 1024.0f));

    // Two clouds to compose in the same world
    GaussianCloud cloud;
    GaussianCloud cloud2;
    GaussianCloud merged;
    cloud.initShaders();
    cloud2.initShaders();
    merged.initShaders();

    DataLoader loader;

    IGFD::FileDialogConfig config;
    bool windowHovered = false;

    // Cloud 1
    bool openDepthDialog1 = false;
    bool openColorDialog1 = false;
    std::string selectedDepthPath1;
    std::string selectedColorPath1;

    // Cloud 2
    bool openDepthDialog2 = false;
    bool openColorDialog2 = false;
    std::string selectedDepthPath2;
    std::string selectedColorPath2;

    // Fixed paths for RGBD images
    const std::string fixedDepthPath1 = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam0\\depth\\frame_000040.png";
    const std::string fixedColorPath1 = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam0\\color\\frame_000040.png";
    const std::string fixedDepthPath2 = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam2\\depth\\frame_000040.png";
    const std::string fixedColorPath2 = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam2\\color\\frame_000040.png";

    glm::mat3 rgbToWorldR1 = {
    0.99902f, -0.0146442f, -0.0417634f,
    0.00912435f, 0.99151f, -0.129354f,
    0.0433062f, 0.128846f, 0.990672f
    };
    glm::vec3 rgbToWorldT1 = {
       255.531f, 162.51f, -1341.79f
    };
    rgbToWorldT1 = rgbToWorldR1 * rgbToWorldT1; // Camera coordinates to world

    glm::mat3 rgbToWorldR2 = {
        -0.997941f, -0.00938262f, 0.0634527f,
        -8.14641e-05f, 0.989427f, 0.145024f,
        -0.0641427f, 0.14472f, -0.98739f
    };
    glm::vec3 rgbToWorldT2 = {
        -132.151f, 189.306f, -1376.99f
    };
    rgbToWorldT2 = rgbToWorldR2 * rgbToWorldT2; // Camera coordinates to world

    // Shared intrinsics (adjust to your sensors)
    glm::mat3 DepthIntrinsics1 = glm::mat3(
        503.272f, 0.0f, 0.0f,
        0.0f, 503.428f, 0.0f,
        311.493f, 341.854f, 1.0f
    );
    glm::mat3 RGBIntrinsics1 = glm::mat3(
        916.106f, 0.0f, 0.0f,
        0.0f, 915.931f, 0.0f,
        959.972f, 545.488f, 1.0f
    );

    glm::mat3 DepthIntrinsics2 = glm::mat3(
        503.263f, 0.0f, 0.0f,
        0.0f, 503.417f, 0.0f,
        324.749f, 336.353f, 1.0f
    );
    glm::mat3 RGBIntrinsics2 = glm::mat3(
        907.692f, 0.0f, 0.0f,
        0.0f, 907.511f, 0.0f,
        957.761f, 551.799f, 1.0f
    );

    // Rotation and Translation matrix from depth to RGB camera
    glm::mat3 R_Cam1 = glm::mat3(
        0.999983f, -0.00586679f, 0.000380531f,
        0.00587709f, 0.995844f, -0.0908823f,
        0.000154238f, 0.090883f, 0.995862f
    );
    // Translation vector from depth to RGB camera (in mm)
    glm::vec3 T_Cam1 = glm::vec3(
        -31.9808f,
        -2.14291f,
        4.06966f
    );

    // Rotation matrix from depth to RGB camera
    glm::mat3 R_Cam2 = glm::mat3(
        0.999992f, -0.00382051f, 0.00112496f,
        0.0039048f, 0.9961f, -0.0881453,
        -0.000783808f, 0.088149f, 0.996107f
    );

    // Translation vector from depth to RGB camera (in mm)
    glm::vec3 T_Cam2 = glm::vec3(
        -32.0719f,
        -2.0198f,
        4.02698f
    );

    // ============================================
    // Third camera parameters for ground truth export
    // ============================================
    // Third camera RGB intrinsics (replace with your actual values)
    glm::mat3 RGBIntrinsics3 = glm::mat3(
        609.147f, 0.0f, 0.0f,
        0.0f, 609.155f, 0.0f,
        633.681f, 362.512f, 1.0f
    );

    // Third camera extrinsics: world-to-camera rotation and translation
    glm::mat3 rgbToWorldR3 = glm::mat3(
        0.818993f, 0.0140402f, -0.573631f,
        0.016921f, 0.998675f, 0.0486023f,
        0.573554f, -0.0495114f, 0.81767f
    );

    glm::vec3 rgbToWorldT3 = glm::vec3(
        351.823f,
        448.301f,
        -1477.75f
    );
    rgbToWorldT3 = rgbToWorldR3 * rgbToWorldT3;

    // Export settings
    int exportWidth = 1280;
    int exportHeight = 720;
    std::string exportPath = "C:\\Users\\b25.jun\\Desktop\\dataset\\results";

    // ============================================
    // Video Playback State
    // ============================================
    RgbdFrameSequence seq1, seq2;
    bool sequencesDiscovered = false;
    bool useGpuLoad = true;

    // Base directories for frame sequences
    const std::string basePath1_depth = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam0\\depth";
    const std::string basePath1_color = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam0\\color";
    const std::string basePath2_depth = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam2\\depth";
    const std::string basePath2_color = "C:\\Users\\b25.jun\\Desktop\\dataset\\experiment-data\\cap1\\cam2\\color";

    // Playback timing
    double lastPlaybackTime = glfwGetTime();
    int playbackFrame = 0;
    bool isPlaying = false;
    float playbackFps = 30.0f;
    bool loopPlayback = true;

    // Rebuild merged on demand
    auto rebuildMerged = [&]() {
        try {
            // Check memory before merge
            size_t free_mem, total_mem;
            cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
            if (err != cudaSuccess) {
                std::cerr << "CUDA Memory check failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }

            size_t required_mem = 0;
            if (cloud.initialized) required_mem += cloud.num_gaussians * 1000; // rough estimate
            if (cloud2.initialized) required_mem += cloud2.num_gaussians * 1000;

            if (cloud.initialized && cloud2.initialized) {
                PointCloudLoader::merge(merged, cloud, cloud2, true);
            }
            else if (cloud.initialized) {
                PointCloudLoader::merge(merged, cloud, GaussianCloud(), true);
            }
            else if (cloud2.initialized) {
                PointCloudLoader::merge(merged, cloud2, GaussianCloud(), true);
            }
            else {
                merged.initialized = false;
                merged.num_gaussians = 0;
            }

            cudaError_t cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                std::cerr << "CUDA error after merge: " << cudaGetErrorString(cerr) << std::endl;
                merged.initialized = false;
                // optionally return / skip any GL usage that depends on 'merged'
            }
            cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess) {
                std::cerr << "cudaGetLastError() after merge: " << cudaGetErrorString(last) << std::endl;
                merged.initialized = false;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception during merge: " << e.what() << std::endl;
        }
        };

    while (!glfwWindowShouldClose(this->w)) {
        // Check for GL context loss at the start of each frame
        GLenum err = glGetError();
        if (err == GL_CONTEXT_LOST) {
            std::cerr << "FATAL: OpenGL context lost! Application cannot continue." << std::endl;
            break;
        }
        else if (err != GL_NO_ERROR) {
            std::cerr << "OpenGL error detected at frame start: " << err << std::endl;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::BeginMainMenuBar();
        GLIntrospection::inspectObjects();
        CudaIntrospection::inspectBuffers();
        ImGui::EndMainMenuBar();

        ImGui::Begin("Window");
        //        ImGui::ShowMetricsWindow();

        if (ImGui::Button("Reload Shaders")) {
            shaderLoader.checkForFileUpdates();
        }

        if (ImGui::Button("Load random")) {
            PointCloudLoader::loadRdm(cloud, 100000, true);
        }
        ImGui::Separator();
        ImGui::Text("RGBD Cloud 1");
        if (ImGui::Button("Load RGBD (Cloud 1)")) {
            config.path = "C:\\Users\\b25.jun\\Desktop";
            config.flags = ImGuiFileDialogFlags_Default;
            ImGuiFileDialog::Instance()->OpenDialog("ChooseDepthDlgKey1", "Choose Depth Image", ".png,.jpg,.jpeg,.tif,.tiff", config);
            openDepthDialog1 = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Fixed Path (Cloud 1)")) {
            try {
                PointCloudLoader::loadRgbd(
                    cloud,
                    fixedDepthPath1,
                    fixedColorPath1,
                    DepthIntrinsics1,
                    RGBIntrinsics1,
                    R_Cam1,
                    T_Cam1,
                    rgbToWorldR1,
                    rgbToWorldT1,
                    true
                );
                selectedDepthPath1 = fixedDepthPath1;
                selectedColorPath1 = fixedColorPath1;

                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error after loading cloud1: " << cudaGetErrorString(err) << std::endl;
                    cloud.initialized = false;
                }
                else {
                    rebuildMerged();
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception loading cloud1 from fixed path: " << e.what() << std::endl;
                cloud.initialized = false;
            }
        }
        if (!selectedDepthPath1.empty()) ImGui::Text("Depth1: %s", selectedDepthPath1.c_str());
        if (!selectedColorPath1.empty()) ImGui::Text("Color1: %s", selectedColorPath1.c_str());

        ImGui::Separator();
        ImGui::Text("RGBD Cloud 2");
        if (ImGui::Button("Load RGBD (Cloud 2)")) {
            config.path = "C:\\Users\\b25.jun\\Desktop";
            config.flags = ImGuiFileDialogFlags_Default;
            ImGuiFileDialog::Instance()->OpenDialog("ChooseDepthDlgKey2", "Choose Depth Image", ".png,.jpg,.jpeg,.tif,.tiff", config);
            openDepthDialog2 = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Fixed Path (Cloud 2)")) {
            try {
                PointCloudLoader::loadRgbd(
                    cloud2,
                    fixedDepthPath2,
                    fixedColorPath2,
                    DepthIntrinsics2,
                    RGBIntrinsics2,
                    R_Cam2,
                    T_Cam2,
                    rgbToWorldR2,
                    rgbToWorldT2,
                    true
                );
                selectedDepthPath2 = fixedDepthPath2;
                selectedColorPath2 = fixedColorPath2;

                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error after loading cloud2: " << cudaGetErrorString(err) << std::endl;
                    cloud2.initialized = false;
                }
                else {
                    rebuildMerged();
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception loading cloud2 from fixed path: " << e.what() << std::endl;
                cloud2.initialized = false;
            }
        }
        if (!selectedDepthPath2.empty()) ImGui::Text("Depth2: %s", selectedDepthPath2.c_str());
        if (!selectedColorPath2.empty()) ImGui::Text("Color2: %s", selectedColorPath2.c_str());

        // ============================================
        // Export Section for Ground Truth Evaluation
        // ============================================
        ImGui::Separator();
        ImGui::Text("Export Rendered Image (Camera 3)");
        
        if (ImGui::Button("Export Render at Camera 3 Pose")) {
            if (merged.initialized) {
                try {
                    std::string exportQuadPath = exportPath + "\\quad_render.png";
                    std::string exportPointCloudPath = exportPath + "\\pointcloud_render.png";
                    
                    merged.exportRenderAtPose(
                        RGBIntrinsics3,
                        rgbToWorldR3,
						rgbToWorldT3,
                        exportWidth,
                        exportHeight,
                        exportQuadPath,
                        true
                    );
                    
                    merged.exportRenderAtPose(
                        RGBIntrinsics3,
                        rgbToWorldR3,
                        rgbToWorldT3,
                        exportWidth,
                        exportHeight,
                        exportPointCloudPath,
                        false
                    );

                    std::cout << "Export completed: " << exportPath << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "Export failed: " << e.what() << std::endl;
                }
            }
            else {
                std::cerr << "No merged cloud to export. Load clouds first." << std::endl;
            }
        }

        // ============================================
        // Video Playback Controls
        // ============================================
        ImGui::Separator();
        ImGui::Text("Video Playback");

        ImGui::Checkbox("Use GPU Loading", &useGpuLoad);

        if (!sequencesDiscovered) {
            if (ImGui::Button("Discover Frame Sequences")) {
                seq1 = PointCloudLoader::discoverFrameSequence(basePath1_depth, basePath1_color);
                seq2 = PointCloudLoader::discoverFrameSequence(basePath2_depth, basePath2_color);
                sequencesDiscovered = true;
                playbackFrame = 0;
                int minFrames = std::min(seq1.totalFrames, seq2.totalFrames);
                std::cout << "Sequences discovered: " << minFrames << " common frames" << std::endl;
            }
        }

        if (sequencesDiscovered) {
            int maxFrames = std::max(seq1.totalFrames, seq2.totalFrames);
            int minFrames = std::min(seq1.totalFrames, seq2.totalFrames);
            ImGui::Text("Cam0: %d frames, Cam2: %d frames", seq1.totalFrames, seq2.totalFrames);

            // Playback controls
            if (isPlaying) {
                if (ImGui::Button("Pause")) {
                    isPlaying = false;
                }
            } else {
                if (ImGui::Button("Play")) {
                    isPlaying = true;
                    lastPlaybackTime = glfwGetTime();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                isPlaying = false;
                playbackFrame = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button("|<")) {
                playbackFrame = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button("<")) {
                playbackFrame = std::max(0, playbackFrame - 1);
            }
            ImGui::SameLine();
            if (ImGui::Button(">")) {
                playbackFrame = std::min(minFrames - 1, playbackFrame + 1);
            }
            ImGui::SameLine();
            if (ImGui::Button(">|")) {
                playbackFrame = minFrames - 1;
            }

            ImGui::SliderInt("Frame", &playbackFrame, 0, minFrames - 1);
            ImGui::SliderFloat("FPS", &playbackFps, 1.0f, 60.0f, "%.1f");
            ImGui::Checkbox("Loop", &loopPlayback);

            ImGui::Text("Current Frame: %d / %d", playbackFrame, minFrames);

            // Load single frame button
            if (ImGui::Button("Load Current Frame")) {
                try {
                    if (playbackFrame < seq1.totalFrames) {
                        if (useGpuLoad) {
                            PointCloudLoader::loadRgbdGpu(cloud, seq1.depthPaths[playbackFrame], seq1.colorPaths[playbackFrame],
                                DepthIntrinsics1, RGBIntrinsics1, R_Cam1, T_Cam1, rgbToWorldR1, rgbToWorldT1, true);
                        } else {
                            PointCloudLoader::loadRgbd(cloud, seq1.depthPaths[playbackFrame], seq1.colorPaths[playbackFrame],
                                DepthIntrinsics1, RGBIntrinsics1, R_Cam1, T_Cam1, rgbToWorldR1, rgbToWorldT1, true);
                        }
                    }
                    if (playbackFrame < seq2.totalFrames) {
                        if (useGpuLoad) {
                            PointCloudLoader::loadRgbdGpu(cloud2, seq2.depthPaths[playbackFrame], seq2.colorPaths[playbackFrame],
                                DepthIntrinsics2, RGBIntrinsics2, R_Cam2, T_Cam2, rgbToWorldR2, rgbToWorldT2, true);
                        } else {
                            PointCloudLoader::loadRgbd(cloud2, seq2.depthPaths[playbackFrame], seq2.colorPaths[playbackFrame],
                                DepthIntrinsics2, RGBIntrinsics2, R_Cam2, T_Cam2, rgbToWorldR2, rgbToWorldT2, true);
                        }
                    }
                    rebuildMerged();
                }
                catch (const std::exception& e) {
                    std::cerr << "Error loading frame " << playbackFrame << ": " << e.what() << std::endl;
                }
            }

            // Auto-advance during playback
            if (isPlaying) {
                double currentTime = glfwGetTime();
                double frameInterval = 1.0 / (double)playbackFps;
                if (currentTime - lastPlaybackTime >= frameInterval) {
                    lastPlaybackTime = currentTime;

                    try {
                        if (playbackFrame < seq1.totalFrames) {
                            if (useGpuLoad) {
                                PointCloudLoader::loadRgbdGpu(cloud, seq1.depthPaths[playbackFrame], seq1.colorPaths[playbackFrame],
                                    DepthIntrinsics1, RGBIntrinsics1, R_Cam1, T_Cam1, rgbToWorldR1, rgbToWorldT1, true);
                            } else {
                                PointCloudLoader::loadRgbd(cloud, seq1.depthPaths[playbackFrame], seq1.colorPaths[playbackFrame],
                                    DepthIntrinsics1, RGBIntrinsics1, R_Cam1, T_Cam1, rgbToWorldR1, rgbToWorldT1, true);
                            }
                        }
                        if (playbackFrame < seq2.totalFrames) {
                            if (useGpuLoad) {
                                PointCloudLoader::loadRgbdGpu(cloud2, seq2.depthPaths[playbackFrame], seq2.colorPaths[playbackFrame],
                                    DepthIntrinsics2, RGBIntrinsics2, R_Cam2, T_Cam2, rgbToWorldR2, rgbToWorldT2, true);
                            } else {
                                PointCloudLoader::loadRgbd(cloud2, seq2.depthPaths[playbackFrame], seq2.colorPaths[playbackFrame],
                                    DepthIntrinsics2, RGBIntrinsics2, R_Cam2, T_Cam2, rgbToWorldR2, rgbToWorldT2, true);
                            }
                        }
                        rebuildMerged();
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error loading frame " << playbackFrame << ": " << e.what() << std::endl;
                        isPlaying = false;
                    }

                    // Advance frame
                    playbackFrame++;
                    if (playbackFrame >= minFrames) {
                        if (loopPlayback) {
                            playbackFrame = 0;
                        } else {
                            playbackFrame = minFrames - 1;
                            isPlaying = false;
                        }
                    }
                }
            }
        }

        // Handle dialogs for Cloud 1
        if (openDepthDialog1) {
            if (ImGuiFileDialog::Instance()->Display("ChooseDepthDlgKey1")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    selectedDepthPath1 = ImGuiFileDialog::Instance()->GetFilePathName();
                    config.path = ImGuiFileDialog::Instance()->GetCurrentPath();
                    ImGuiFileDialog::Instance()->Close();
                    ImGuiFileDialog::Instance()->OpenDialog("ChooseColorDlgKey1", "Choose Color Image", ".png,.jpg,.jpeg,.tif,.tiff", config);
                    openDepthDialog1 = false;
                    openColorDialog1 = true;
                }
                else {
                    ImGuiFileDialog::Instance()->Close();
                    openDepthDialog1 = false;
                }
            }
        }
        if (openColorDialog1) {
            if (ImGuiFileDialog::Instance()->Display("ChooseColorDlgKey1")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    selectedColorPath1 = ImGuiFileDialog::Instance()->GetFilePathName();
                    ImGuiFileDialog::Instance()->Close();
                    openColorDialog1 = false;

                    if (!selectedDepthPath1.empty() && !selectedColorPath1.empty()) {
                        // Load the first RGBD set into cloud (world pose rgbCamToWorld1)
                        try {
                            PointCloudLoader::loadRgbd(
                                cloud,
                                selectedDepthPath1,
                                selectedColorPath1,
                                DepthIntrinsics1,
                                RGBIntrinsics1,
                                R_Cam1,
                                T_Cam1,
                                rgbToWorldR1,
                                rgbToWorldT1,
                                true
                            );

                            // Synchronize to catch any CUDA errors immediately
                            cudaError_t err = cudaDeviceSynchronize();
                            if (err != cudaSuccess) {
                                std::cerr << "CUDA error after loading cloud1: " << cudaGetErrorString(err) << std::endl;
                                cloud.initialized = false;
                            }
                            else {
                                rebuildMerged();
                            }
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Exception loading cloud1: " << e.what() << std::endl;
                            cloud.initialized = false;
                        }
                    }
                }
                else {
                    ImGuiFileDialog::Instance()->Close();
                    openColorDialog1 = false;
                }
            }
        }

        // Handle dialogs for Cloud 2
        if (openDepthDialog2) {
            if (ImGuiFileDialog::Instance()->Display("ChooseDepthDlgKey2")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    selectedDepthPath2 = ImGuiFileDialog::Instance()->GetFilePathName();
                    config.path = ImGuiFileDialog::Instance()->GetCurrentPath();
                    ImGuiFileDialog::Instance()->Close();
                    ImGuiFileDialog::Instance()->OpenDialog("ChooseColorDlgKey2", "Choose Color Image", ".png,.jpg,.jpeg,.tif,.tiff", config);
                    openDepthDialog2 = false;
                    openColorDialog2 = true;
                }
                else {
                    ImGuiFileDialog::Instance()->Close();
                    openDepthDialog2 = false;
                }
            }
        }
        if (openColorDialog2) {
            if (ImGuiFileDialog::Instance()->Display("ChooseColorDlgKey2")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    selectedColorPath2 = ImGuiFileDialog::Instance()->GetFilePathName();
                    ImGuiFileDialog::Instance()->Close();
                    openColorDialog2 = false;

                    if (!selectedDepthPath2.empty() && !selectedColorPath2.empty()) {
                        // Load the second RGBD set into cloud2 (world pose rgbCamToWorld2)
                        try {
                            PointCloudLoader::loadRgbd(
                                cloud2,
                                selectedDepthPath2,
                                selectedColorPath2,
                                DepthIntrinsics2,
                                RGBIntrinsics2,
                                R_Cam2,
                                T_Cam2,
                                rgbToWorldR2,
                                rgbToWorldT2,
                                true
                            );

                            // Synchronize to catch any CUDA errors immediately
                            cudaError_t err = cudaDeviceSynchronize();
                            if (err != cudaSuccess) {
                                std::cerr << "CUDA error after loading cloud2: " << cudaGetErrorString(err) << std::endl;
                                cloud2.initialized = false;
                            }
                            else {
                                rebuildMerged();
                            }
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Exception loading cloud2: " << e.what() << std::endl;
                            cloud2.initialized = false;
                        }
                    }
                }
                else {
                    ImGuiFileDialog::Instance()->Close();
                    openColorDialog2 = false;
                }
            }
        }

        // Info
        if (merged.initialized) {
            ImGui::Text("Merged: %d gaussians.", merged.num_gaussians);
        }
        else {
            if (cloud.initialized) ImGui::Text("Cloud1: %d", cloud.num_gaussians);
            if (cloud2.initialized) ImGui::Text("Cloud2: %d", cloud2.num_gaussians);
        }

        camera.updateView(w, windowHovered, (float)scroll);

        int width, height;
        glfwGetWindowSize(w, &width, &height);
        glfwGetFramebufferSize(w, &width, &height);

        // Check for errors before rendering
        GLenum pre_render_err = glGetError();
        if (pre_render_err == GL_CONTEXT_LOST) {
            std::cerr << "GL_CONTEXT_LOST detected before rendering!" << std::endl;
            break;
        }

        glViewport(0, 0, width, height); // reset the viewport
        err = glGetError();
        if (err == GL_CONTEXT_LOST) {
            std::cerr << "GL_CONTEXT_LOST after glViewport!" << std::endl;
            break;
        }

        //        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                // need to clear with alpha = 1 for front to back blending
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        err = glGetError();
        if (err == GL_CONTEXT_LOST) {
            std::cerr << "GL_CONTEXT_LOST after glClearColor!" << std::endl;
            break;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        err = glGetError();
        if (err == GL_CONTEXT_LOST) {
            std::cerr << "GL_CONTEXT_LOST after glClear!" << std::endl;
            break;
        }
        //        windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

        // Only merged view rendering (points or quads)
        if (merged.initialized) {
            try {
                merged.render(camera);
                err = glGetError();
                if (err == GL_CONTEXT_LOST) {
                    std::cerr << "GL_CONTEXT_LOST after merged.render()!" << std::endl;
                    break;
                }
                merged.GUI(camera);
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during rendering: " << e.what() << std::endl;
            }
        }

        windowHovered = ImGui::GetIO().WantCaptureMouse;

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        scroll *= 0.90f;

        glfwSwapBuffers(this->w);
        glfwPollEvents();

        // Only sleep if not playing video (to maintain real-time playback)
        if (!isPlaying) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (windowHovered) {
            scroll = 0;
        }

    }
}