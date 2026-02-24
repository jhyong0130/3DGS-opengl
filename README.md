# 3DGS-OpenGL

On the fly 3D Gaussian Splatting renderer built with OpenGL and CUDA.

## Prerequisites

- [Visual Studio 2022](https://visualstudio.microsoft.com/) with the **Desktop development with C++** workload
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x (SM 8.6 target by default)

Library dependencies (glfw3, OpenCV) are declared in `vcpkg.json` and installed automatically by Visual Studio's built-in vcpkg during CMake configure.

## Building on Windows

1. Open the project folder in Visual Studio 2022 (**File > Open > Folder**).
2. Visual Studio will detect `CMakeSettings.json`, run vcpkg to install dependencies, and configure the project automatically.
3. Select the **x64-Debug** configuration from the toolbar dropdown.
4. Set **3DGS_OpenGL.exe** as the startup item (dropdown next to the green play button).
5. Copy the `resources/` folder into the build output directory (e.g. `out/build/x64-Debug/`). The shaders are loaded at runtime relative to the working directory and are not copied automatically by the build.

The application opens an interactive window. Use the ImGui panels to load PLY files or RGBD image sequences.

## Project structure

```
├── CMakeLists.txt              # Build configuration
├── vcpkg.json                  # Dependency manifest
├── src/
│   ├── Main.cpp                # Entry point
│   ├── Window.cu               # Main render loop and GUI
│   ├── GaussianCloud.cpp/h     # Gaussian splat data and rendering
│   ├── PointCloudLoader.cpp/h  # PLY / RGBD / random cloud loading
│   ├── DataLoader.cpp/h        # Multi-view dataset loading
│   ├── Sort.cu/cuh             # CUDA radix sort
│   ├── RgbdLoadCuda.cu/cuh     # CUDA RGBD unprojection kernels
│   ├── RenderingBase/          # OpenGL & CUDA utility classes
│   ├── imgui/                  # ImGui (vendored)
│   ├── glad/                   # OpenGL loader (vendored)
│   ├── stb/                    # stb_image (vendored)
│   └── miniply/                # PLY parser (vendored)
├── resources/
│   └── shaders/                # GLSL vertex, fragment, and compute shaders
└── CMakeSettings.json          # Visual Studio CMake config
```
