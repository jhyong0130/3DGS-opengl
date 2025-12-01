#pragma once

#include <imgui/imgui.h>

class GUI2D {
private:
    ImGuiIO& io;
    float _fps = 0.f;

public:
    bool drawPoints        = true;  // Flag to control whether to draw balls in the render
    bool drawCVT        = true;  // Flag to control whether to draw balls in the render
    bool drawXPlane        = true; 
    bool drawYPlane        = false; 
    bool drawZPlane        = false; 

    bool mousePoint        = false;
    bool resetCam         = false;
    bool shrink         = false;

    bool upsample = false;
    bool hide_symmetry = false;
    
    int nbrIter = 0;
    int nbrIter_todo = 0;
    int nbrPoints = 0;
    int nbpoints = 10000;
    float step = 0.f;
    float pointSize = 5.0f;
    float GauSize = 0.5; //0.004f;
    float Scale = 20.0f;
    float XPlane = 0.0f;
    float YPlane = 0.0f;
    float ZPlane = 0.0f;

    GUI2D()
        : io(ImGui::GetIO()) {};

    void render2D();
    
    inline void set_fps(float fps) {_fps = fps;}
};
