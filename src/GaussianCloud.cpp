#include "GaussianCloud.h"
#include "RenderingBase/VAO.h"

#include "imgui/imgui.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include <opencv2/opencv.hpp>

#include "../resources/shaders/common/CommonTypes.h"

using namespace glm;

const GLenum FBO_FORMAT = GL_RGBA16F;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    }

float h_SDF_Torus(vec3 point, float R_max, float R_min) {
    float qx = sqrt(point.x * point.x + point.y * point.y) - R_max;
    float qz = point.z;
    return sqrt(qx * qx + qz * qz) - R_min;
}

// -- Add helper functions
void checkGLResetAndError(const std::string& tag) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) { 
        std::cerr << "[GL ERROR] " << tag << " glGetError() = 0x" << std::hex << err << std::dec << std::endl; 
    } 
    GLenum reset = glGetGraphicsResetStatus();
    if (reset != GL_NO_ERROR) { 
        std::cerr << "[GL RESET] " << tag << " glGetGraphicsResetStatus() = 0x" << std::hex << reset << std::dec << std::endl; 
    }
}

void Print(Uniforms uniforms) {
    std::cout << "scale modifier: " << uniforms.scale_modifier << std::endl;
    std::cout << "size: " << uniforms.width << ", " << uniforms.width << std::endl;
    std::cout << "focal: " << uniforms.focal_x << ", " << uniforms.focal_y << std::endl;
    std::cout << "near_far plane: " << uniforms.near_plane << ", " << uniforms.far_plane << std::endl;
    std::cout << "min_opacity: " << uniforms.min_opacity << std::endl;
    std::cout << "selected_gaussian: " << uniforms.selected_gaussian << std::endl;
    std::cout << "antialiasing: " << uniforms.antialiasing << std::endl;
    

    std::cout << "viewMat: " << uniforms.viewMat[0][0] << ", " << uniforms.viewMat[0][1] << ", " << uniforms.viewMat[0][2] << ", " << uniforms.viewMat[0][3] << std::endl;
                    std::cout << uniforms.viewMat[1][0] << ", " << uniforms.viewMat[1][1] << ", " << uniforms.viewMat[1][2] << ", " << uniforms.viewMat[1][3] << std::endl;
                    std::cout << uniforms.viewMat[2][0] << ", " << uniforms.viewMat[2][1] << ", " << uniforms.viewMat[2][2] << ", " << uniforms.viewMat[2][3] << std::endl;
                    std::cout << uniforms.viewMat[3][0] << ", " << uniforms.viewMat[3][1] << ", " << uniforms.viewMat[3][2] << ", " << uniforms.viewMat[3][3] << std::endl;

    std::cout << "projMat: " << uniforms.projMat[0][0] << ", " << uniforms.projMat[0][1] << ", " << uniforms.projMat[0][2] << ", " << uniforms.projMat[0][3] << std::endl;
                    std::cout << uniforms.projMat[1][0] << ", " << uniforms.projMat[1][1] << ", " << uniforms.projMat[1][2] << ", " << uniforms.projMat[1][3] << std::endl;
                    std::cout << uniforms.projMat[2][0] << ", " << uniforms.projMat[2][1] << ", " << uniforms.projMat[2][2] << ", " << uniforms.projMat[2][3] << std::endl;
                    std::cout << uniforms.projMat[3][0] << ", " << uniforms.projMat[3][1] << ", " << uniforms.projMat[3][2] << ", " << uniforms.projMat[3][3] << std::endl;

                    
}

// Copy OpenGL texture to cv::Mat and show with OpenCV
void showGLTextureInOpenCV(GLuint textureID, int width, int height) {
    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Allocate buffer to read texture
    std::vector<unsigned char> pixels(width * height * 4); // 4 channels (RGBA)

    // Read texture from GPU
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Create cv::Mat from raw data
    cv::Mat image(height, width, CV_8UC4, pixels.data());

    cv::imwrite("debug_texture.png", image);

    // OpenCV uses BGR, so convert
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);

    // Flip vertically if needed (OpenGL's origin is bottom-left)
    //cv::flip(image, image, 0);

    // Show the image
    //cv::imshow("OpenGL Texture", image);
    //cv::waitKey(1);
}

void GaussianCloud::prepareRender(Camera &camera, bool GT) {
    // Check if OpenGL context is valid before proceeding
    GLenum contextError = glGetError();
    if (contextError == GL_CONTEXT_LOST) {
        std::cerr << "OpenGL context lost before prepareRender. Cannot proceed." << std::endl;
        return;
    }

    const float width = (float)camera.getFramebufferSize().x;
    const float height = (float)camera.getFramebufferSize().y;

    const GLenum formats[] = { GL_RGBA8, GL_RGBA16F, GL_RGBA32F };

    if (!GT && (width != fbo.getWidth() || height != fbo.getHeight())) {
        fbo.init(width, height);
        fbo.createAttachment(GL_COLOR_ATTACHMENT0, FBO_FORMAT, GL_RGBA, GL_FLOAT);
        fbo.drawBuffersAllAttachments();
        if (!fbo.checkComplete()) {
            exit(0);
        }

        emptyfbo.init(width, height);
        emptyfbo.makeEmpty();
        if (!emptyfbo.checkComplete()) {
            exit(0);
        }
    }

    checkGLResetAndError("before storing visible gaussian");
    const int zero = 0;
    visible_gaussians_counter.storeData(&zero, 1, sizeof(int), 0, false, false, true);

	// compute camera parameters
    glm::mat3 R = glm::mat3(camera.getViewMatrix());
    glm::vec3 T = glm::vec3(camera.getViewMatrix()[3]);

    float a = camera.getProjectionMatrix()[0][0];
    float b = camera.getProjectionMatrix()[1][1];

    glViewport(0, 0, width, height);

    glm::mat3 K = glm::transpose(glm::mat3(
        a * width / 2.0f, 0, width / 2.0f,
        0, b * height / 2.0f, height / 2.0f,
        0, 0, 1.0f));
    
    glm::mat3 R_GL_TO_CV = glm::mat3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1);

    R = R_GL_TO_CV * R;
    T = R_GL_TO_CV * T;

    //const mat4 rot = glm::rotate(mat4(1.0f), radians(180.0f), vec3(1, 0, 0));

    Uniforms uniforms_cpu = {};
    uniforms_cpu.viewMat = camera.getViewMatrix();// *rot;
    uniforms_cpu.projMat = camera.getProjectionMatrix();

    uniforms_cpu.camera_pos = vec4(camera.getPosition(), 1.0f);
    uniforms_cpu.K = glm::mat4(K);  // Convert mat3 to mat4
    uniforms_cpu.R = glm::mat4(R);
    uniforms_cpu.T = glm::vec4(T, 0.0f);
    //std::cout << uniforms_cpu.camera_pos.x << ", " << uniforms_cpu.camera_pos.y << ", " << uniforms_cpu.camera_pos.z << std::endl;
        
    uniforms_cpu.num_gaussians = num_gaussians;
    uniforms_cpu.near_plane = camera.getNearPlane();
    uniforms_cpu.far_plane = camera.getFarPlane();
    //std::cout << uniforms_cpu.near_plane << ", " << uniforms_cpu.far_plane << std::endl;
    
    uniforms_cpu.scale_modifier = scale_modifier;
	// uniforms_cpu.scale_neus = scale_neus;

    uniforms_cpu.selected_gaussian = selected_gaussian;
    uniforms_cpu.min_opacity = min_opacity;
    uniforms_cpu.width = width;
    uniforms_cpu.height = height;

    auto fov2focal = [](float fov, float pixels){
        return pixels / (2.0f * tan(fov / 2.0f));
    };

    uniforms_cpu.focal_x = fov2focal(camera.getFovX(), width);
    uniforms_cpu.focal_y = fov2focal(camera.getFovY(), height);
    //std::cout << uniforms_cpu.focal_x << ", " << uniforms_cpu.focal_y << std::endl;

    uniforms_cpu.antialiasing = int(antialiasing);
    uniforms_cpu.front_to_back = int(front_to_back);
    uniforms_cpu.SDF_scale = SDF_scale;

    uniforms_cpu.positions = reinterpret_cast<vec4 *>(positions.getGLptr());
    uniforms_cpu.normals = reinterpret_cast<vec4*>(normals.getGLptr());
    uniforms_cpu.covX = reinterpret_cast<vec4*>(covariance[0].getGLptr());
    uniforms_cpu.covY = reinterpret_cast<vec4*>(covariance[1].getGLptr());
    uniforms_cpu.covZ = reinterpret_cast<vec4*>(covariance[2].getGLptr());
    //uniforms_cpu.rotations = reinterpret_cast<vec4 *>(rotations.getGLptr());
    //uniforms_cpu.scales = reinterpret_cast<vec4 *>(scales.getGLptr());
    uniforms_cpu.sdf = reinterpret_cast<float *>(sdf.getGLptr());
    uniforms_cpu.scale_neus = reinterpret_cast<float*>(scale_neus.getGLptr());
    uniforms_cpu.sh_coeffs_red = reinterpret_cast<float *>(sh_coeffs[0].getGLptr());
    uniforms_cpu.sh_coeffs_green = reinterpret_cast<float *>(sh_coeffs[1].getGLptr());
    uniforms_cpu.sh_coeffs_blue = reinterpret_cast<float *>(sh_coeffs[2].getGLptr());

    uniforms_cpu.visible_gaussians_counter = reinterpret_cast<int *>(visible_gaussians_counter.getGLptr());
    uniforms_cpu.gaussians_depth = reinterpret_cast<float *>(gaussians_depths.getGLptr());
    uniforms_cpu.gaussians_indices = reinterpret_cast<int *>(gaussians_indices.getGLptr());
    uniforms_cpu.sorted_depths = reinterpret_cast<float *>(sorted_depths.getGLptr());
    uniforms_cpu.sorted_gaussian_indices = reinterpret_cast<int *>(sorted_gaussian_indices.getGLptr());

    uniforms_cpu.bounding_boxes = reinterpret_cast<vec4 *>(bounding_boxes.getGLptr());
    uniforms_cpu.conic_opacity = reinterpret_cast<vec4 *>(conic_opacity.getGLptr());
    uniforms_cpu.eigen_vecs = reinterpret_cast<vec2 *>(eigen_vecs.getGLptr());
    uniforms_cpu.predicted_colors = reinterpret_cast<vec4 *>(predicted_colors.getGLptr());

    checkGLResetAndError("before uniforms.storeData");
    uniforms.storeData(&uniforms_cpu, 1, sizeof(Uniforms));
    checkGLResetAndError("after uniforms.storeData");
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniforms.getID());
}


void GaussianCloud::render(Camera &camera) {

    prepareRender(camera);
    checkGLResetAndError("after preparing render");

    if(renderAsQuads){
        fbo.bind();
        glViewport(0, 0, fbo.getWidth(), fbo.getHeight());
        // need to clear with alpha = 1 for front to back blending
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
		// Always do front to back blending
        glBlendEquation(GL_FUNC_ADD);
        glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);

        {
            auto& q = timers[OPERATIONS::TEST_VISIBILITY].push_back();
            q.begin();
            // cull non-visible gaussians
            testVisibilityShader.start();
            glDispatchCompute((num_gaussians+127)/128, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            testVisibilityShader.stop();
            q.end();
            glCopyNamedBufferSubData(visible_gaussians_counter.getID(), counter.getID(), 0, 0, sizeof(int));
        }

        // read back the number of visible gaussians. That's a cpu / gpu synchronization, but it's ok.
        num_visible_gaussians = *(int*)glMapNamedBuffer(counter.getID(), GL_READ_ONLY);
        glUnmapNamedBuffer(counter.getID());

        // sort the gaussians by depth
        {
            auto& q = timers[OPERATIONS::SORT].push_back();
            q.begin();
            sort.sort(gaussians_depths, sorted_depths, gaussians_indices, sorted_gaussian_indices, num_visible_gaussians);
            q.end();
        }

        {
            auto& q = timers[OPERATIONS::COMPUTE_BOUNDING_BOXES].push_back();
            q.begin();
            computeBoundingBoxesShader.start();
            glDispatchCompute((num_visible_gaussians+127)/128, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            computeBoundingBoxesShader.stop();
            q.end();
        }

        // --- DEBUG: read back some computed bounding boxes ---
        //if (num_visible_gaussians > 0) {
        //    glFinish(); // ensure compute finished
        //
        //    const int debug_count = num_visible_gaussians;
        //    // Each bounding box is vec4(center_x, center_y, half_w, half_h) in pixels
        //    std::vector<float> boxes = bounding_boxes.getAsFloats(debug_count * 4);
        //    // conic_opacity is vec4(conic.x, conic.y, conic.z, opacity)
        //    std::vector<float> conics = conic_opacity.getAsFloats(debug_count * 4);
        //    // Read sorted gaussian indices to get original gaussian IDs
        //    std::vector<int> sorted_indices_cpu(debug_count);
        //    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_gaussian_indices.getID());
        //    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, debug_count * sizeof(int), sorted_indices_cpu.data());
        //    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        //
        //    // Read positions and covariances for the visible gaussians
        //    std::vector<float> pos_data = positions.getAsFloats(num_gaussians * 4);
        //    std::vector<float> covX_data = covariance[0].getAsFloats(num_gaussians * 4);
        //    std::vector<float> covY_data = covariance[1].getAsFloats(num_gaussians * 4);
        //    std::vector<float> covZ_data = covariance[2].getAsFloats(num_gaussians * 4);
        //    std::vector<float> normal_data = normals.getAsFloats(num_gaussians * 4);
        //
        //    std::cout << "\n[ALPHA DEBUG] num_visible_gaussians = " << num_visible_gaussians << "\n";
        //    std::cout << "Camera pos: (" << camera.getPosition().x << ", " 
        //              << camera.getPosition().y << ", " << camera.getPosition().z << ")\n";
        //
        //    const int num_samples = 10;
        //    const int step = std::max(1, debug_count / num_samples);
        //
        //    for (int i = 0; i < debug_count && i < num_samples * step; i += step) {
        //        int gid = sorted_indices_cpu[i]; // original gaussian ID
        //        
        //        // Bounding box
        //        float cx = boxes[i * 4 + 0];
        //        float cy = boxes[i * 4 + 1];
        //        float hw = boxes[i * 4 + 2];
        //        float hh = boxes[i * 4 + 3];
        //
        //        // Conic and opacity
        //        float conic_x = conics[i * 4 + 0];
        //        float conic_y = conics[i * 4 + 1];
        //        float conic_z = conics[i * 4 + 2];
        //        float opacity = conics[i * 4 + 3];
        //
        //        // Position
        //        float px = pos_data[gid * 4 + 0];
        //        float py = pos_data[gid * 4 + 1];
        //        float pz = pos_data[gid * 4 + 2];
        //
        //        // Normal
        //        float nx = normal_data[gid * 4 + 0];
        //        float ny = normal_data[gid * 4 + 1];
        //        float nz = normal_data[gid * 4 + 2];
        //
        //        // Covariance diagonal (variance)
        //        float var_x = covX_data[gid * 4 + 0];
        //        float var_y = covY_data[gid * 4 + 1];
        //        float var_z = covZ_data[gid * 4 + 2];
        //
        //        // Compute determinant of covariance matrix
        //        float cov_det = covX_data[gid * 4 + 0] * (covY_data[gid * 4 + 1] * covZ_data[gid * 4 + 2] - covY_data[gid * 4 + 2] * covZ_data[gid * 4 + 1])
        //                      - covX_data[gid * 4 + 1] * (covY_data[gid * 4 + 0] * covZ_data[gid * 4 + 2] - covY_data[gid * 4 + 2] * covZ_data[gid * 4 + 0])
        //                      + covX_data[gid * 4 + 2] * (covY_data[gid * 4 + 0] * covZ_data[gid * 4 + 1] - covY_data[gid * 4 + 1] * covZ_data[gid * 4 + 0]);
        //
        //        // Compute distance from camera to gaussian
        //        glm::vec3 cam_pos = camera.getPosition();
        //        float dist = sqrt((px - cam_pos.x) * (px - cam_pos.x) + 
        //                          (py - cam_pos.y) * (py - cam_pos.y) + 
        //                          (pz - cam_pos.z) * (pz - cam_pos.z));
        //
        //        std::cout << "  [" << i << "] GaussianID=" << gid << "\n"
        //                  << "    Position: (" << px << ", " << py << ", " << pz << ") dist=" << dist << " mm\n"
        //                  << "    Normal: (" << nx << ", " << ny << ", " << nz << ")\n"
        //                  << "    Cov diag (var): (" << var_x << ", " << var_y << ", " << var_z << ") det=" << cov_det << "\n"
        //                  << "    BBox center: (" << cx << ", " << cy << ") half_size: (" << hw << ", " << hh << ")\n"
        //                  << "    Conic: (" << conic_x << ", " << conic_y << ", " << conic_z << ") opacity=" << opacity << "\n";
        //    }
        //    std::cout << std::endl;
        //}

        {
            auto& q = timers[OPERATIONS::PREDICT_COLORS_VISIBLE].push_back();
            q.begin();
            // Evaluate the sh basis only for the visible gaussians
            // Groups of 128 threads, with 16 threads working together on the same gaussian: 8*16 = 128
            predictColorsShader.start();
            glDispatchCompute((num_visible_gaussians+7)/8, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            predictColorsShader.stop();
            q.end();
        }

        if(selected_gaussian != -1){
            glFinish();
            auto box = bounding_boxes.getAsFloats(4);
            auto conic = conic_opacity.getAsFloats(4);
            auto eigen_vec = eigen_vecs.getAsFloats(2);

            ImGui::Text("Bounding box: %.1f %.1f %.1f %.1f", box[0], box[1], box[2], box[3]);
            ImGui::Text("conic_opacity: %.4f %.4f %.4f %.2f", conic[0], conic[1], conic[2], conic[3]);
            ImGui::Text("eigen_vec: %.2f %.2f", eigen_vec[0], eigen_vec[1]);
        }

        {
            auto& q = timers[OPERATIONS::DRAW_AS_QUADS].push_back();
            q.begin();

            // draw a 2D quad for every visible gaussian
            auto& s = softwareBlending ? quad_interlock_Shader : quadShader;
            s.start();

            VAO vao; // empty vertex array
            vao.bind();
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glDrawArrays(GL_TRIANGLES, 0, num_visible_gaussians * 6);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            vao.unbind();

            s.stop();
            q.end();
        }

        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);

        {
            auto& q = timers[OPERATIONS::BLIT_FBO].push_back();
            q.begin();
            fbo.blit(0, GL_COLOR_BUFFER_BIT);
            q.end();
        }

        if (softwareBlending) {
            emptyfbo.unbind();
        }
        else {
            fbo.unbind();
        }
    }

    if(renderAsPoints) {
        glEnable(GL_DEPTH_TEST);

        {
            // Predict colors for all the gaussians
            auto& q = timers[OPERATIONS::PREDICT_COLORS_ALL].push_back();
            q.begin();
            predictColorsForAllShader.start();
            // Groups of 128 threads, with 16 threads working together on the same gaussian: 8*16 = 128
            glDispatchCompute((num_gaussians+7)/8, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            predictColorsForAllShader.stop();
            q.end();
        }
        checkGLResetAndError("after predicting colors");
        {
            auto& q = timers[OPERATIONS::DRAW_AS_POINTS].push_back();
            q.begin();
            // Draw as a point cloud
            glEnable(GL_PROGRAM_POINT_SIZE);
            pointShader.start();
            VAO vao; // empty vertex array
            vao.bind();
            glDrawArrays(GL_POINTS, 0, num_gaussians);
            vao.unbind();
            pointShader.stop();
            q.end();
            glDisable(GL_PROGRAM_POINT_SIZE);
        }
        checkGLResetAndError("after predicting colors");
    }

}

void GaussianCloud::initShaders() {
    pointShader.init_uniforms({});
    testVisibilityShader.init_uniforms({});
    computeBoundingBoxesShader.init_uniforms({});
    quadShader.init_uniforms({});
    quad_interlock_Shader.init_uniforms({});
    predictColorsShader.init_uniforms({});
    predictColorsForAllShader.init_uniforms({});

    counter.storeData(nullptr, 1, sizeof(int), GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT, false, true, true);
}

void GaussianCloud::freeRawCudaBuffers() {
}

void GaussianCloud::clearCpuData() {
    positions_cpu.clear();   positions_cpu.shrink_to_fit();
    normals_cpu.clear();     normals_cpu.shrink_to_fit();
    covX_cpu.clear();        covX_cpu.shrink_to_fit();
    covY_cpu.clear();        covY_cpu.shrink_to_fit();
    covZ_cpu.clear();        covZ_cpu.shrink_to_fit();
    scales_cpu.clear();      scales_cpu.shrink_to_fit();
    rotations_cpu.clear();   rotations_cpu.shrink_to_fit();
    opacities_cpu.clear();   opacities_cpu.shrink_to_fit();
    sdf_cpu.clear();         sdf_cpu.shrink_to_fit();
    scale_neus_cpu.clear();  scale_neus_cpu.shrink_to_fit();
}

void GaussianCloud::GUI(Camera& camera) {
    ImGui::PushID(this);
    const float frac = num_visible_gaussians / float(num_gaussians) * 100.0f;
    ImGui::Text("There are %d currently visible gaussians (%.1f%%).", num_visible_gaussians, frac);

    ImGui::Checkbox("Render as points", &renderAsPoints);
    ImGui::Checkbox("Render as quads", &renderAsQuads);
    ImGui::Checkbox("Antialiasing", &antialiasing);
    ImGui::Checkbox("Front to back blending", &front_to_back);
    ImGui::Checkbox("Software alpha-blending", &softwareBlending);
    ImGui::SliderFloat("scale_modifier", &scale_modifier, 0.001f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
	// ImGui::SliderFloat("scale_neus", &scale_neus, 1.0f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("min_opacity", &min_opacity, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);


    if(selected_gaussian >= 0 && selected_gaussian < num_gaussians){
        ImGui::Text("Position: %.3f %.3f %.3f %.3f", positions_cpu[selected_gaussian].x, positions_cpu[selected_gaussian].y, positions_cpu[selected_gaussian].z, positions_cpu[selected_gaussian].w);
        ImGui::Text("Scale: %.3f %.3f %.3f %.3f", scales_cpu[selected_gaussian].x, scales_cpu[selected_gaussian].y, scales_cpu[selected_gaussian].z, scales_cpu[selected_gaussian].w);
        ImGui::Text("Rotation: %.3f %.3f %.3f %.3f", rotations_cpu[selected_gaussian].x, rotations_cpu[selected_gaussian].y, rotations_cpu[selected_gaussian].z, rotations_cpu[selected_gaussian].w);
        ImGui::Text("Opacity: %.3f", opacities_cpu[selected_gaussian]);
    }

    if(ImGui::TreeNode("Timers")){
        if(renderAsPoints){
            ImGui::Text("Point rendering:");
            ImGui::Text("Predict colors: %.3fms", timers[OPERATIONS::PREDICT_COLORS_ALL].getLastResult() * 1.0E-6);
            ImGui::Text("Draw points: %.3fms", timers[OPERATIONS::DRAW_AS_POINTS].getLastResult() * 1.0E-6);
            ImGui::Separator();
        }
        if(renderAsQuads){
            ImGui::Text("Quad rendering:");
            ImGui::Text("Test visibility: %.3fms", timers[OPERATIONS::TEST_VISIBILITY].getLastResult() * 1.0E-6);
            ImGui::Text("Sort: %.3fms", timers[OPERATIONS::SORT].getLastResult() * 1.0E-6);
            ImGui::Text("Compute bounding boxes: %.3fms", timers[OPERATIONS::COMPUTE_BOUNDING_BOXES].getLastResult() * 1.0E-6);
            ImGui::Text("Predict colors: %.3fms", timers[OPERATIONS::PREDICT_COLORS_VISIBLE].getLastResult() * 1.0E-6);
            ImGui::Text("Draw quads: %.3fms", timers[OPERATIONS::DRAW_AS_QUADS].getLastResult() * 1.0E-6);
            ImGui::Separator();
        }
        ImGui::Separator();

        ImGui::TreePop();
    }

    ImGui::Text("Rendering Info:");
    ImGui::Text("Visible Gaussians: %d / %d", num_visible_gaussians, num_gaussians);

    // Show covariance statistics
    if (ImGui::CollapsingHeader("Covariance Debug")) {
        static int sample_idx = 0;
        static bool cov_loaded = false;
        static float covX[4] = { 0, 0, 0, 0 };
        static float covY[4] = { 0, 0, 0, 0 };
        static float covZ[4] = { 0, 0, 0, 0 };
        ImGui::SliderInt("Sample Index", &sample_idx, 0, num_gaussians - 1);

        if (sample_idx < num_gaussians && !positions_cpu.empty()) {
            ImGui::Text("Position: (%.3f, %.3f, %.3f)",
                positions_cpu[sample_idx].x,
                positions_cpu[sample_idx].y,
                positions_cpu[sample_idx].z);
            ImGui::Text("Depth (z): %.3f", positions_cpu[sample_idx].z);
            ImGui::Text("Opacity: %.3f", opacities_cpu[sample_idx]);

            // Read covariance matrix from GPU
            if (ImGui::Button("Read Covariance")) {
                glFinish(); // Ensure GPU operations are complete

                // Read entire buffer (or at least up to sample_idx + 1)
                int elements_to_read = (sample_idx + 1) * 4; // 4 floats per element
                auto covX_all = covariance[0].getAsFloats(elements_to_read);
                auto covY_all = covariance[1].getAsFloats(elements_to_read);
                auto covZ_all = covariance[2].getAsFloats(elements_to_read);

                // Extract the values for the selected gaussian
                int offset = sample_idx * 4;
                covX[0] = covX_all[offset];
                covX[1] = covX_all[offset + 1];
                covX[2] = covX_all[offset + 2];
                covX[3] = covX_all[offset + 3];

                covY[0] = covY_all[offset];
                covY[1] = covY_all[offset + 1];
                covY[2] = covY_all[offset + 2];
                covY[3] = covY_all[offset + 3];

                covZ[0] = covZ_all[offset];
                covZ[1] = covZ_all[offset + 1];
                covZ[2] = covZ_all[offset + 2];
                covZ[3] = covZ_all[offset + 3];

                cov_loaded = true;
            }

            // Display covariance data if it has been loaded
            if (cov_loaded) {
                ImGui::Separator(); 
                ImGui::Text("Covariance Matrix:");
                ImGui::Text("  [%.6f, %.6f, %.6f]", covX[0], covX[1], covX[2]);
                ImGui::Text("  [%.6f, %.6f, %.6f]", covY[0], covY[1], covY[2]);
                ImGui::Text("  [%.6f, %.6f, %.6f]", covZ[0], covZ[1], covZ[2]);
            }
        }
    }
    ImGui::PopID();
}

// Export a rendered image at a specific camera pose
void GaussianCloud::exportRenderAtPose(
    const glm::mat3& intrinsics,       // Camera intrinsic matrix K (OpenCV style)
    const glm::mat3& R_cam_to_world,   // Rotation: camera to world
    const glm::vec3& T_cam_to_world,   // Translation: camera position in world
    int width, int height,
    const std::string& outputPath,
    bool useQuadRendering)             // true = quads, false = point cloud
{
    std::cout << "=== Export Render at Pose ===" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Render mode: " << (useQuadRendering ? "Quads" : "Point Cloud") << std::endl;
    std::cout << "Camera position (input): " << T_cam_to_world.x << ", " 
              << T_cam_to_world.y << ", " << T_cam_to_world.z << std::endl;
    
    // Extract intrinsics to projMat
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[2][0];
    float cy = intrinsics[2][1];
    
    std::cout << "Intrinsics: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
    
    Camera exportCamera;
    exportCamera.setFrameBufferSize(width, height);
    
    // Build projection matrix manually (matching Camera::setProjectionMatrix logic)
    float nearPlane = 10.0f;
    float farPlane = 5000.0f;
    
    glm::mat4 projMat(0.0f);
    projMat[0][0] = 2.0f * fx / float(width);
    projMat[1][1] = 2.0f * fy / float(height);

    // Correct mapping:
    //   x_ndc offset:  2*cx/w - 1
    //   y_ndc offset:  1 - 2*cy/h   (because OpenCV v grows downward)
    projMat[2][0] = 2.0f * cx / float(width) - 1.0f;
    projMat[2][1] = 2.0f * cy / float(height) - 1.0f;

    projMat[2][2] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    projMat[2][3] = -1.0f;
    projMat[3][2] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);

    // Extrisics to viewMat
    // F: OpenCV camera axes -> OpenGL camera axes
    glm::mat3 F(1.0f);
    F[1][1] = -1.0f;
    F[2][2] = -1.0f;
    glm::mat3 R_cam_to_world_GL = F * R_cam_to_world * F;
    glm::vec3 T_cam_to_world_GL = F * T_cam_to_world;
    glm::mat3 R_wc = glm::transpose(R_cam_to_world_GL);  // world-to-camera rotation
    glm::vec3 T_wc = -R_wc * T_cam_to_world_GL;

    // Pack to mat4 (column-major)
    glm::mat4 viewMatrix(1.0f);
    viewMatrix[0] = glm::vec4(R_wc[0], 0.0f);
    viewMatrix[1] = glm::vec4(R_wc[1], 0.0f);
    viewMatrix[2] = glm::vec4(R_wc[2], 0.0f);
    viewMatrix[3] = glm::vec4(T_wc, 1.0f);

    std::cout << "View Matrix:" << std::endl;
    for (int row = 0; row < 4; row++) {
        std::cout << "  ";
        for (int col = 0; col < 4; col++) {
            std::cout << viewMatrix[col][row] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compute camera position from view matrix for verification
    glm::mat3 R_from_view = glm::mat3(viewMatrix);
    glm::vec3 t_from_view = glm::vec3(viewMatrix[3]);
    glm::vec3 camPos = -glm::transpose(R_from_view) * t_from_view;
    
    std::cout << "Camera pos (extracted): " << camPos.x << ", " << camPos.y << ", " << camPos.z << std::endl;
    
    FBO exportFbo;
    exportFbo.init(width, height);
    exportFbo.createAttachment(GL_COLOR_ATTACHMENT0, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    
    // Use renderbuffer for depth - doesn't need CUDA interop
    GLuint depthRbo;
    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glNamedFramebufferRenderbuffer(exportFbo.getID(), GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRbo);
    
    exportFbo.drawBuffersAllAttachments();
    if (!exportFbo.checkComplete()) {
        std::cerr << "Failed to create export FBO" << std::endl;
        glDeleteRenderbuffers(1, &depthRbo);
        return;
    }
    
    exportFbo.bind();
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Reset visible counter
    const int zero = 0;
    visible_gaussians_counter.storeData(&zero, 1, sizeof(int), 0, false, false, true);
    
    // Build uniforms manually
    glm::mat3 R_uni = glm::mat3(viewMatrix);
    glm::vec3 T_uni = glm::vec3(viewMatrix[3]);
    
    float a = projMat[0][0];
    float b = projMat[1][1];
    
    glm::mat3 K_uni = glm::transpose(glm::mat3(
        a * width / 2.0f, 0, width / 2.0f,
        0, b * height / 2.0f, height / 2.0f,
        0, 0, 1.0f));
    
    glm::mat3 R_GL_TO_CV = glm::mat3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1);
    
    R_uni = R_GL_TO_CV * R_uni;
    T_uni = R_GL_TO_CV * T_uni;
    
    Uniforms uniforms_cpu = {};
    uniforms_cpu.viewMat = viewMatrix;
    uniforms_cpu.projMat = projMat;
    uniforms_cpu.camera_pos = glm::vec4(camPos, 1.0f);
    uniforms_cpu.K = glm::mat4(K_uni);
    uniforms_cpu.R = glm::mat4(R_uni);
    uniforms_cpu.T = glm::vec4(T_uni, 0.0f);
    uniforms_cpu.num_gaussians = num_gaussians;
    uniforms_cpu.near_plane = nearPlane;
    uniforms_cpu.far_plane = farPlane;
    uniforms_cpu.scale_modifier = scale_modifier;
    // uniforms_cpu.scale_neus = scale_neus;
    uniforms_cpu.selected_gaussian = -1;
    uniforms_cpu.min_opacity = min_opacity;
    uniforms_cpu.width = width;
    uniforms_cpu.height = height;
    uniforms_cpu.focal_x = fx;
    uniforms_cpu.focal_y = fy;
    uniforms_cpu.antialiasing = int(antialiasing);
    uniforms_cpu.front_to_back = int(front_to_back);
    uniforms_cpu.SDF_scale = SDF_scale;
    
    // Buffer pointers
    uniforms_cpu.positions = reinterpret_cast<glm::vec4*>(positions.getGLptr());
    uniforms_cpu.normals = reinterpret_cast<glm::vec4*>(normals.getGLptr());
    uniforms_cpu.covX = reinterpret_cast<glm::vec4*>(covariance[0].getGLptr());
    uniforms_cpu.covY = reinterpret_cast<glm::vec4*>(covariance[1].getGLptr());
    uniforms_cpu.covZ = reinterpret_cast<glm::vec4*>(covariance[2].getGLptr());
    uniforms_cpu.sdf = reinterpret_cast<float*>(sdf.getGLptr());
	uniforms_cpu.scale_neus = reinterpret_cast<float*>(scale_neus.getGLptr());
    uniforms_cpu.sh_coeffs_red = reinterpret_cast<float*>(sh_coeffs[0].getGLptr());
    uniforms_cpu.sh_coeffs_green = reinterpret_cast<float*>(sh_coeffs[1].getGLptr());
    uniforms_cpu.sh_coeffs_blue = reinterpret_cast<float*>(sh_coeffs[2].getGLptr());
    uniforms_cpu.visible_gaussians_counter = reinterpret_cast<int*>(visible_gaussians_counter.getGLptr());
    uniforms_cpu.gaussians_depth = reinterpret_cast<float*>(gaussians_depths.getGLptr());
    uniforms_cpu.gaussians_indices = reinterpret_cast<int*>(gaussians_indices.getGLptr());
    uniforms_cpu.sorted_depths = reinterpret_cast<float*>(sorted_depths.getGLptr());
    uniforms_cpu.sorted_gaussian_indices = reinterpret_cast<int*>(sorted_gaussian_indices.getGLptr());
    uniforms_cpu.bounding_boxes = reinterpret_cast<glm::vec4*>(bounding_boxes.getGLptr());
    uniforms_cpu.conic_opacity = reinterpret_cast<glm::vec4*>(conic_opacity.getGLptr());
    uniforms_cpu.eigen_vecs = reinterpret_cast<glm::vec2*>(eigen_vecs.getGLptr());
    uniforms_cpu.predicted_colors = reinterpret_cast<glm::vec4*>(predicted_colors.getGLptr());
    
    uniforms.storeData(&uniforms_cpu, 1, sizeof(Uniforms));
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniforms.getID());
    
    if (useQuadRendering) {
        // === QUAD RENDERING ===
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        
        // Test visibility
        testVisibilityShader.start();
        glDispatchCompute((num_gaussians + 127) / 128, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        testVisibilityShader.stop();
        glCopyNamedBufferSubData(visible_gaussians_counter.getID(), counter.getID(), 0, 0, sizeof(int));
        
        num_visible_gaussians = *(int*)glMapNamedBuffer(counter.getID(), GL_READ_ONLY);
        glUnmapNamedBuffer(counter.getID());
        
        std::cout << "Export (Quads): " << num_visible_gaussians << " / " << num_gaussians << " visible" << std::endl;
        
        if (num_visible_gaussians == 0) {
            std::cerr << "WARNING: No visible gaussians!" << std::endl;
            exportFbo.unbind();
            return;
        }
        
        // Sort by depth
        sort.sort(gaussians_depths, sorted_depths, gaussians_indices, sorted_gaussian_indices, num_visible_gaussians);
        
        // Compute bounding boxes
        computeBoundingBoxesShader.start();
        glDispatchCompute((num_visible_gaussians + 127) / 128, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        computeBoundingBoxesShader.stop();
        
        // Predict colors for visible gaussians
        predictColorsShader.start();
        glDispatchCompute((num_visible_gaussians + 7) / 8, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        predictColorsShader.stop();
        
        // Draw quads
        auto& s = softwareBlending ? quad_interlock_Shader : quadShader;
        s.start();
        VAO vao;
        vao.bind();
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDrawArrays(GL_TRIANGLES, 0, num_visible_gaussians * 6);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        vao.unbind();
        s.stop();
        
        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);
    }
    else {
        // === POINT CLOUD RENDERING ===
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        
        // Predict colors for all gaussians
        predictColorsForAllShader.start();
        glDispatchCompute((num_gaussians + 7) / 8, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        predictColorsForAllShader.stop();
        
        std::cout << "Export (Points): rendering " << num_gaussians << " points" << std::endl;
        
        // Draw as point cloud
        glEnable(GL_PROGRAM_POINT_SIZE);
        pointShader.start();
        VAO vao;
        vao.bind();
        glDrawArrays(GL_POINTS, 0, num_gaussians);
        vao.unbind();
        pointShader.stop();
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
    
    glFinish();
    
    // Read back
    std::vector<float> pixels_float(width * height * 4);
    glBindTexture(GL_TEXTURE_2D, exportFbo.getAttachment(GL_COLOR_ATTACHMENT0)->getID());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels_float.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    exportFbo.unbind();
    
    std::vector<unsigned char> pixels(width * height * 4);
    for (int i = 0; i < width * height * 4; i++) {
        pixels[i] = static_cast<unsigned char>(glm::clamp(pixels_float[i] * 255.0f, 0.0f, 255.0f));
    }
    
    cv::Mat image(height, width, CV_8UC4, pixels.data());
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    cv::flip(image, image, 0);
    
    cv::imwrite(outputPath, image);
    std::cout << "Exported to: " << outputPath << std::endl;
}