cmake_minimum_required(VERSION 3.20.0)

project(CVTSPLAT LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(ZLIB_USE_STATIC_LIBS "ON")

find_package(glfw3 CONFIG REQUIRED)
find_package(assimp REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})

get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("${importTargetsAfter}")

message("Current Generator is  \"${CMAKE_GENERATOR}\"")

set(Utils_files
	# glad, to load opengl functions
	src/glad/gl.c

	# stb, to read and write png, jpeg, hdr files
	src/stb/stb_image.c

	src/miniply/miniply.cpp

	# Imgui
	src/imgui/imgui.cpp 
	src/imgui/imgui_draw.cpp 
	src/imgui/imgui_tables.cpp	
	src/imgui/imgui_widgets.cpp 
	src/imgui/imgui_impl_glfw.cpp 
	src/imgui/imgui_impl_opengl3.cpp
	src/imgui/ImGuiFileDialog.cpp

)

# Basic utilities for opengl objects and cuda
set(RenderingBase_files
		src/RenderingBase/GLIntrospection.cpp
		src/RenderingBase/CudaGLInterop.cpp
		src/RenderingBase/GLBuffer.cpp
		src/RenderingBase/FBO.cpp
		src/RenderingBase/VAO.cpp
		src/RenderingBase/SharedContext.cpp
		src/RenderingBase/AsyncWorkers.cpp
		src/RenderingBase/Shader.cpp
		src/RenderingBase/GLShaderLoader.cpp
		src/RenderingBase/Camera.cpp
        src/RenderingBase/Texture2D.cpp
		src/RenderingBase/MeshLoader.cpp
		src/RenderingBase/CudaBuffer.cuh
		src/RenderingBase/CudaTimer.cu
		src/RenderingBase/CudaBuffer.cu
		src/RenderingBase/CudaIntrospection.cu
		src/RenderingBase/CudaIntrospection.cuh
		src/Window.cu
)

# Force Window.cu to be compiled with the host C++ compiler (MSVC) rather than nvcc.
# This avoids nvcc attempting to parse Boost headers (Boost.Fusion uses modern template features)
# which can cause template/type parsing errors when nvcc front-end is used.
set_source_files_properties(src/Window.cu PROPERTIES LANGUAGE CXX)

# Main program
add_executable(CVTSPLAT
	src/Main.cpp
	src/Window.cu
	${Utils_files} ${RenderingBase_files}
		src/PointCloudLoader.cpp
		src/PointCloudLoader.h
		src/GaussianCloud.cpp
        src/GaussianCloud.h
		src/DataLoader.cpp
		src/DataLoader.h
		src/Sort.cu
		src/Sort.cuh
		src/Backward.cu
		src/Backward.cuh
		src/adam_cuda.cu
		src/adam_cuda.cuh
		src/CVTupdate.cu
		src/CVTupdate.cuh
		src/KNN_cuda.cu
		src/KNN_cuda.cuh
		src/knn_cu/lbvh_kernels.cu
		src/knn_cu/query_knn_kernels.cu
)
target_link_libraries(CVTSPLAT glfw assimp::assimp CGAL::CGAL Boost::boost ${OpenCV_LIBS})
target_compile_options(CVTSPLAT PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
		--use_fast_math
		--ptxas-options=-v
		--resource-usage
#		-G
#		-g
#		-rdc=true
		-Xcudafe="--diag_suppress=20012"
		>)

#set_source_files_properties(src/Main.cu PROPERTIES COMPILE_FLAGS "-G -g")

set_target_properties(CVTSPLAT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# IMGUI test program
add_executable(IMGUI_test src/imgui/main.cpp src/imgui/imgui_demo.cpp ${Utils_files})
target_link_libraries(IMGUI_test glfw ${CMAKE_DL_LIBS})

INSTALL(TARGETS CVTSPLAT IMGUI_test DESTINATION ${MyProject_SOURCE_DIR}/install)

message("CUDA_LIBRARIES is  \"${CUDA_LIBRARIES}\"")